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
fold boundaries of rolling cross-validation. ``DLModel`` is the PyTorch
variant: one cross-section of symbols per training step, each with its own
window of past bars, and ``.pth`` checkpoints. ``MLModel`` is the numpy variant for tree models and
other libraries that do their own early stopping, with ``.joblib``
checkpoints. Concrete heads live in ``quantlab/dl_model`` and
``quantlab/ml_model``.
"""

import copy
import dataclasses
import json
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
from tqdm import tqdm

from quantlab.backend import XrBackend
from quantlab.base.data import InsufficientHistoryError
from quantlab.base.stopping import StoppingRule
from quantlab.dl_model.training import CrossSectionWindows, TargetTransform
from quantlab.enums.constant import Date
from quantlab.ml_model.backend import MlBackend
from quantlab.utils.atomic import write_json_atomically
from quantlab.utils.jsonable import to_jsonable
from quantlab.utils.metrics import regression_panel_metrics
from quantlab.utils.symbol_axis import sort_symbol_axis
from quantlab.utils.split import purge_segments
from quantlab.utils.timer import Timer

from .config import DLConfig, MLConfig


class BaseModel(ABC):
    """Framework-agnostic base class of every model head.

    A head is configured with a list of factor objects (its features) and
    a list of label objects (its targets). ``collect()`` pulls both into one
    panel indexed by ``(timestamp, symbol)``; ``train()`` fits the head on the
    ``train_*`` dates of its config and writes a checkpoint; ``load()`` restores
    one; ``predict()`` and ``predict_panel()`` run inference. Those public entry
    points are implemented once, here, and are not overridden by any head.

    The training framework is the job of the two variants. ``DLModel`` is the
    torch variant and ``MLModel`` the numpy variant; each declares two plain
    class attributes that satisfy the abstract properties below:

    ``config_cls`` is the config class the variant accepts. The ``config``
    setter checks it first, and the config loader reads it from the class
    before creating an instance, so it must be a class attribute.
    ``checkpoint_suffix`` is the checkpoint file suffix. ``train`` and
    ``train_cv`` use it to name files, and ``load()`` uses it to reject a
    file of the wrong kind before building any model.

    Checkpoints are written under ``config.model_save_dir`` as
    ``{class}_trial_{timestamp}/{experiment}/{experiment}{suffix}``, with a
    ``config.json`` sidecar file next to the checkpoint. The sidecar holds
    the full config and a record of what the head was trained on.

    Parameters
    ----------
    config : DLConfig or MLConfig
        The model configuration. It must be an instance of the variant's
        ``config_cls``.

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
    Given a head ``MyHead`` (a subclass of ``MLModel`` or ``DLModel``), a
    factor object exposing variables ``f_a`` and ``f_b``, and a label
    object exposing ``ret``::

        >>> model = MyHead(MLConfig(
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

    def __init__(self, config: DLConfig | MLConfig):
        """Initialize the model; see the class docstring for parameters."""
        self.config = config
        self._set_random_seed(self.config.random_seed)

        self.model = None
        # The backtester checks the checkpoint before collecting data and
        # `load()` checks it again; remembering warnings logs each only once.
        self._emitted_load_warnings: set[str] = set()

        self.data_backend = XrBackend()
        self._wandb_recorder: wandb.sdk.wandb_run.Run = None  # type: ignore

    @property
    @abstractmethod
    def config_cls(self) -> type:
        """The config class this variant accepts.

        Concrete variants satisfy it with a plain class attribute.

        Examples
        --------
        >>> DLModel.config_cls
        <class 'quantlab.base.config.DLConfig'>
        """

    @property
    @abstractmethod
    def checkpoint_suffix(self) -> str:
        """The checkpoint file suffix, including the leading dot.

        Concrete variants satisfy it with a plain class attribute.

        Examples
        --------
        >>> MLModel.checkpoint_suffix
        '.joblib'
        """

    @staticmethod
    def _set_random_seed(seed: int):
        """Seed the framework-agnostic generators: ``random`` and numpy.

        ``DLModel._set_random_seed`` adds the torch seeds on top of this.
        """
        random.seed(seed)
        np.random.seed(seed)

    def __repr__(self) -> str:
        """Return ``ClassName(config=...)``."""
        return f"{self.__class__.__name__}(config={self.config})"

    @property
    def config(self) -> DLConfig | MLConfig:
        """The model's configuration object.

        Examples
        --------
        >>> model.config.model_save_dir
        checkpoints
        """
        return self._config

    @config.setter
    def config(self, config: DLConfig | MLConfig):
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
        >>> model.config = MLConfig(factors=[factor], labels=[label],
        ...                         model_save_dir="checkpoints",
        ...                         factor_data_strategy="read",
        ...                         label_data_strategy="read",
        ...                         start_date="2024-01-01")
        >>> model.config.end_date
        '2100-01-01'
        """
        self._config = self._normalize_config(config)

    def _normalize_config(self, config: DLConfig | MLConfig) -> DLConfig | MLConfig:
        """Return ``config`` checked against ``config_cls`` with its defaults filled in.

        A model variant that validates or completes its config overrides
        this, calls ``super()._normalize_config(config)`` first and returns a
        new config built with ``dataclasses.replace``. It reads the config
        it is given, never ``self.config``.

        Parameters
        ----------
        config : DLConfig or MLConfig
            The config to normalise. It is not modified.

        Returns
        -------
        DLConfig or MLConfig
            A new config with ``name``, ``start_date`` and ``end_date`` set.

        Raises
        ------
        TypeError
            If ``config`` is not an instance of ``config_cls``, a factor is
            a label, or a label is not one (see ``_check_roles``).

        Examples
        --------
        >>> model._normalize_config(config).name
        'quantlab.ml_model.xgb.XGBoostRegressor'
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

    def _check_roles(self, config: DLConfig | MLConfig) -> None:
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
        ``DLModel`` returns ``window_bars - 1``.

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
        ['early_stopping', 'early_stopping_patience', 'end_date']
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

        self._assert_trained_variables(p)
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

    def _assert_trained_variables(self, p: Path) -> None:
        """Check the checkpoint's recorded variables against the model's own.

        The feature and label names (and their order) the checkpoint was
        trained on are taken from ``trained_on`` in its ``config.json``. If
        that record is missing, the older ``factors[]`` / ``labels[]``
        ``factor_names`` config fields are used instead; because a user can
        order those freely, they are compared as sets, and a difference in
        order only logs a warning. When neither is available the checkpoint
        is loaded as given, with a warning.

        Only ``config.json`` is read, so the check can run before any data is
        collected. Each distinct warning is logged once per model instance.

        Raises
        ------
        ValueError
            If the recorded and declared variables differ, naming
            the checkpoint and both variable lists.
        """
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

        feats = features[factors].sortby(["timestamp", "symbol"])
        x = self.to_array(feats, factors)
        y = np.asarray(self._predict_panel_array(x), dtype=np.float64)
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

    def _predict_panel_array(self, x: np.ndarray) -> np.ndarray:
        """Variant hook of ``predict_panel``: ``[T, S, F]`` array in, ``[T, S, L]`` out.

        A plain method rather than an abstract one, so that the set of
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

        A new trial directory is created under ``model_save_dir``, a wandb run
        is opened, and the variant's ``_fit`` trains, evaluates and writes the
        checkpoint. The metrics ``_fit`` returns are written to
        ``metrics.json`` beside the checkpoint's ``config.json``, with NaN and
        inf as null; a variant that returns no metrics writes no file.
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
        >>> sorted(json.loads((checkpoint.parent / "metrics.json").read_text()))[:3]
        ['test_ic', 'test_loss', 'test_mae']
        """
        project_name = self._new_project_name()
        experiment_name = f"{self.class_name}_total"
        model_name = f"{experiment_name}{self.checkpoint_suffix}"
        self._init_wandb(
            project_name=project_name,
            experiment_name=experiment_name,
        )
        metrics = self._fit(
            project_name=project_name,
            experiment_name=experiment_name,
            model_name=model_name,
        )
        checkpoint = (
            Path(self.config.model_save_dir) / project_name / experiment_name / model_name
        ).absolute()
        if metrics is not None:
            write_json_atomically(
                checkpoint.parent / self.METRICS_FILENAME,
                to_jsonable(metrics),
                indent=2,
            )
        return checkpoint

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
        ``test_*`` metrics ``_fit`` returned.
        """
        self.config = dataclasses.replace(
            self.config,
            train_start=fold["train_start"],
            train_end=fold["train_end"],
            test_start=fold["test_start"],
            test_end=fold["test_end"],
        )

        experiment_name = f"{self.class_name}_cv_fold_{fold['fold']}"
        model_name = f"{experiment_name}{self.checkpoint_suffix}"

        self._init_wandb(
            project_name=project_name,
            experiment_name=experiment_name,
        )
        metrics = self._fit(
            project_name=project_name,
            experiment_name=experiment_name,
            model_name=model_name,
        )
        return {
            **record,
            "experiment_name": experiment_name,
            "checkpoint": str(
                (
                    Path(self.config.model_save_dir)
                    / project_name
                    / experiment_name
                    / model_name
                ).absolute()
            ),
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
            the purge leaves a fold no training bar.

        Examples
        --------
        >>> results = model.train_cv(train_periods=20)
        >>> len(results)
        4
        >>> results[0]["fold"], results[0]["checkpoint"].endswith("fold_0.joblib")
        (0, True)
        """
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
    def _fit(
        self, project_name: str, experiment_name: str, model_name: str
    ) -> dict | None:
        """Train, evaluate and save once, then finish the current wandb run.

        The checkpoint goes to ``model_save_dir / project_name /
        experiment_name / model_name``. Returns the metrics of every evaluated
        split as one dict keyed ``{split}_{metric}`` with split ``train``,
        ``val`` (only when there is a validation segment) and ``test``
        (``train`` writes it to ``metrics.json``, ``train_cv`` averages it),
        or None when the variant produces no metrics.
        """

    def _compute_metrics(self, y: np.ndarray, pred: np.ndarray) -> dict:
        """Return ``regression_panel_metrics`` for the first label on raw values.

        ``y`` and ``pred`` are ``[T, S, L]``; only label index 0 is scored.
        """
        return regression_panel_metrics(pred[..., 0], y[..., 0])

    @abstractmethod
    def _predict(self, data: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """Variant implementation of ``predict``; ``self.model`` is guaranteed set."""

    @abstractmethod
    def _write_checkpoint(self, path: Path) -> None:
        """Write ``self.model`` to ``path``; the directory and sidecar already exist."""

    @abstractmethod
    def _read_checkpoint(self, path: Path) -> None:
        """Restore ``self.model`` from ``path``; the suffix is already checked."""


class DLModel(BaseModel):
    """Torch variant: one cross-section per step, trained through per-step hooks.

    A step is one bar (ADR 0006): the symbols with at least one finite
    feature there, each carrying its last ``window_bars`` bars of features.
    The network sees ``[S_t, N, F]``; S_t changes from bar to bar, so it must
    not depend on the order or the number of symbols, and a symbol the model
    never saw in training still gets a prediction.

    The base class owns the data side: the windows (see
    ``CrossSectionWindows``: optional clip to ±3, NaN -> 0, zero rows before
    a symbol's history), the shuffled bar order, the head's target
    transform, the stopping rule, the ``train_*`` / ``val_*`` / ``test_*``
    metrics on the raw first label, ``.pth`` checkpoints and prediction.
    The head owns the learning side, through these hooks:

    ``_init_model(num_features, num_labels, hyperparameters)``
        Build the ``nn.Module``; its ``forward`` maps ``[S_t, N, F]`` to
        ``[S_t, L]`` (prediction relies on that shape).
    ``_init_optim(model)``
        Return the optimizer, kept on ``self.optim``; or None when the head
        updates its parameters by itself.
    ``_train_one_batch(epoch, x, y)``
        One optimisation step on one bar: zero_grad, forward, loss,
        backward, any clipping, step. Returns the loss.
    ``_val_one_batch(epoch, x, y)``
        The validation loss of one bar, under ``no_grad`` in eval mode.
    ``_test_one_batch(epoch, x, y)``
        Called on every test bar after each epoch; heads typically log here.
    ``_preprocess(x)``
        Optional; applied to every window tensor in training and prediction
        alike. The default returns it unchanged.

    ``x`` is ``[S_t, N, F]`` on ``device``; ``y`` is ``[S_t, L]``, the
    head's ``target_transform`` of the bar's labels, with NaN where a label
    is missing. Symbols with a missing label stay in ``x`` as context, so the
    loss must leave the NaN entries out (``masked_mse`` in
    ``quantlab.dl_model.training`` does). A head also declares
    ``window_bars`` (N), ``target_transform`` (a ``TargetTransform``) and
    ``stopping`` (a ``StoppingRule``); it may set ``clip_features = False``.

    The model's warm-up is N - 1 bars: every feature request, in training
    and in a backtest, starts that many bars earlier on each factor's
    dataset calendar, so the first requested bar has a full window.

    Examples
    --------
    A minimal head: a linear map of each symbol's latest bar::

        >>> class LastBarHead(DLModel):
        ...     window_bars = 5
        ...     target_transform = TargetTransform("zscore")
        ...     stopping = ValLossPatience(patience=3)
        ...     def _init_model(self, num_features, num_labels, hyperparameters):
        ...         return LastBarLinear(num_features, num_labels)
        ...     def _init_optim(self, model):
        ...         return torch.optim.Adam(model.parameters(), lr=self.config.lr)
        ...     def _train_one_batch(self, epoch, x, y):
        ...         self.optim.zero_grad()
        ...         loss = masked_mse(self.model(x), y)
        ...         loss.backward()
        ...         self.optim.step()
        ...         return loss.detach()
        ...     def _val_one_batch(self, epoch, x, y):
        ...         return masked_mse(self.model(x), y)
        ...     def _test_one_batch(self, epoch, x, y):
        ...         return masked_mse(self.model(x), y)
        >>> head = LastBarHead(DLConfig(
        ...     factors=[factor], labels=[label], model_save_dir="checkpoints",
        ...     factor_data_strategy="read", label_data_strategy="read",
        ...     epochs=2,
        ...     train_start="2024-01-01", train_end="2024-01-30",
        ...     test_start="2024-01-31", test_end="2024-02-09",
        ... ))
        >>> head.collect().train().suffix
        '.pth'

    where ``LastBarLinear`` is ``nn.Linear(num_features, num_labels)``
    applied to ``x[:, -1]``.
    """

    config_cls = DLConfig
    checkpoint_suffix = ".pth"

    #: Clip features to ±``FEATURE_CLIP`` before NaN becomes 0. A head may
    #: set it to False.
    clip_features: bool = True
    #: Clip bound of ``clip_features``, matching Qlib's ``RobustZScoreNorm``.
    FEATURE_CLIP = 3.0

    @property
    @abstractmethod
    def window_bars(self) -> int:
        """N, the number of bars in each symbol's input window.

        Examples
        --------
        >>> head.window_bars
        5
        """

    @property
    @abstractmethod
    def target_transform(self) -> TargetTransform:
        """How a bar's raw labels become the ``y`` the step hooks receive.

        Examples
        --------
        >>> head.target_transform
        TargetTransform(kind='zscore', drop_extreme=0.0)
        """

    @property
    @abstractmethod
    def stopping(self) -> StoppingRule:
        """When training stops and which weights are kept.

        Examples
        --------
        >>> head.stopping
        ValLossPatience(patience=3)
        """

    @abstractmethod
    def _init_model(
        self, num_features: int, num_labels: int, hyperparameters: dict
    ) -> torch.nn.Module:
        """Build the network; ``forward`` maps ``[S_t, N, F]`` to ``[S_t, L]``.

        The base class moves it to ``device``. ``hyperparameters`` is
        ``config.hyperparameters``, a free-form dict.
        """

    def _init_optim(self, model: torch.nn.Module):
        """Return the optimizer for ``model``, or None if the head updates itself.

        The base class keeps it on ``self.optim`` for the step hooks. The
        default raises ``NotImplementedError``, so every head that trains
        overrides it.
        """
        raise NotImplementedError(f"{self.class_name} does not implement _init_optim")

    @abstractmethod
    def _train_one_batch(
        self, epoch: int, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Run one optimisation step on one bar and return its loss.

        The base class has already called ``model.train()``; the hook
        performs ``zero_grad``, forward, loss, ``backward``, any clipping and
        ``step``. ``x`` is ``[S_t, N, F]`` and ``y`` is ``[S_t, L]`` with NaN
        where a label is missing. The mean of the returned losses over the
        epoch's bars is the epoch's training loss, so it must convert with
        ``float()``.
        """

    @abstractmethod
    def _val_one_batch(
        self, epoch: int, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Return the validation loss of one bar.

        Called in eval mode under ``no_grad``. The mean over the validation
        bars is the epoch's validation loss, which the stopping rule reads,
        and the mean over each split's bars is its ``{split}_loss`` metric,
        so the result must convert with ``float()``.
        """

    @abstractmethod
    def _test_one_batch(
        self, epoch: int, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor | None:
        """Evaluate one test bar; called after each epoch in eval mode without gradients.

        The return value is not used by the base class; heads typically log
        metrics here.
        """

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Transform a ``[S_t, N, F]`` window tensor; shared by training and prediction.

        The windows are already clipped and free of NaN. The default returns
        ``x`` unchanged.
        """
        return x

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

    def _windows(self, x: np.ndarray) -> CrossSectionWindows:
        """Return the cross-section windows over a ``[T, S, F]`` array."""
        return CrossSectionWindows(
            x,
            self.window_bars,
            self.FEATURE_CLIP if self.clip_features else None,
        )

    def _x(self, windows: CrossSectionWindows, t: int, symbols) -> torch.Tensor:
        """Bar ``t``'s preprocessed ``[S_t, N, F]`` windows of ``symbols``, on device."""
        return self._preprocess(
            torch.from_numpy(windows.window(t, symbols)).to(self.device)
        )

    def _steps(
        self, windows: CrossSectionWindows, y: np.ndarray, bars, *, drop: bool
    ):
        """Yield ``(t, x, y)`` for each bar with at least one finite target.

        ``y`` is the transformed target; with ``drop`` the symbols
        ``target_transform.kept`` removes leave the cross-section first.
        """
        transform = self.target_transform
        for t in bars:
            symbols = windows.symbols(t)
            if drop:
                symbols = symbols[transform.kept(y[t, symbols])]
            target = transform.apply(y[t, symbols])
            if not np.isfinite(target).any():
                continue
            yield (
                t,
                self._x(windows, t, symbols),
                torch.from_numpy(target.astype(np.float32)).to(self.device),
            )

    def _train_epoch(
        self,
        epoch: int,
        windows: CrossSectionWindows,
        y: np.ndarray,
        bars: np.ndarray,
        rng: np.random.Generator,
    ) -> float:
        """Call ``_train_one_batch`` on each training bar in shuffled order.

        Returns the mean loss, NaN when no bar has a finite target.
        """
        self.model.train()  # type: ignore[union-attr]
        return self._mean_loss(
            self._train_one_batch(epoch, x, target)
            for _, x, target in self._steps(windows, y, rng.permutation(bars), drop=True)
        )

    @staticmethod
    def _mean_loss(losses) -> float:
        """Mean of the ``float`` of each loss; NaN when there is none."""
        values = [float(loss) for loss in losses]
        return float(np.mean(values)) if values else float("nan")

    def _val_loss(
        self, epoch: int, windows: CrossSectionWindows, y: np.ndarray, bars
    ) -> float:
        """Mean ``_val_one_batch`` over ``bars``, in eval mode without gradients.

        Every symbol of the cross-section is kept. NaN when no bar has a
        finite target.
        """
        self.model.eval()  # type: ignore[union-attr]
        with torch.no_grad():
            return self._mean_loss(
                self._val_one_batch(epoch, x, target)
                for _, x, target in self._steps(windows, y, bars, drop=False)
            )

    def _test_epoch(
        self, epoch: int, windows: CrossSectionWindows, y: np.ndarray, bars
    ) -> None:
        """Call ``_test_one_batch`` on every test bar, in eval mode without gradients."""
        self.model.eval()  # type: ignore[union-attr]
        with torch.no_grad():
            for _, x, target in self._steps(windows, y, bars, drop=False):
                self._test_one_batch(epoch, x, target)

    def _predict_bars(self, windows: CrossSectionWindows, bars) -> np.ndarray:
        """Return ``[len(bars), S, L]`` predictions, NaN outside each cross-section.

        Raises
        ------
        ValueError
            If the network does not return a ``[S_t, num_labels]`` tensor.
        """
        out = np.full(
            (len(bars), windows.present.shape[1], self.num_labels), np.nan
        )
        expected = self.num_labels
        self.model.eval()  # type: ignore[union-attr]
        with torch.no_grad():
            for i, t in enumerate(bars):
                symbols = windows.symbols(t)
                if not len(symbols):
                    continue
                pred = self.model(self._x(windows, t, symbols))  # type: ignore[misc]
                if not isinstance(pred, torch.Tensor) or tuple(pred.shape) != (
                    len(symbols),
                    expected,
                ):
                    got = tuple(pred.shape) if isinstance(pred, torch.Tensor) else type(pred)
                    raise ValueError(
                        f"{self.class_name}: the network must map [S_t, N, F] to "
                        f"a [S_t, L] = {[len(symbols), expected]} tensor, got {got}"
                    )
                out[i, symbols] = pred.detach().cpu().numpy()
        return out

    def _evaluate(
        self,
        epoch: int,
        split: str,
        windows: CrossSectionWindows,
        y: np.ndarray,
        bars: np.ndarray,
    ) -> dict[str, float]:
        """Return one split's metrics and write them to the wandb summary.

        ``{split}_loss`` is the mean ``_val_one_batch`` over the split's bars;
        the other keys are ``_compute_metrics`` on the raw labels, as for
        every model.
        """
        pred = self._predict_bars(windows, bars)
        metrics = {f"{split}_loss": self._val_loss(epoch, windows, y, bars)}
        for key, value in self._compute_metrics(y[bars], pred).items():
            metrics[f"{split}_{key}"] = value
        if self._wandb_recorder is not None:
            self._wandb_recorder.summary.update(metrics)
        return metrics

    def _fit(
        self, project_name: str, experiment_name: str, model_name: str
    ) -> dict:
        """Train bar by bar until the stopping rule fires, then evaluate and save.

        The panel is split by ``_fit_segments``, but every window reads the
        whole collected panel, so the first validation and test bars (and
        the first training bar, through the warm-up) have full windows.
        Each epoch calls ``_train_one_batch`` on the training bars in
        shuffled order, then ``_val_one_batch`` on the validation bars when
        there are any, then ``_test_one_batch`` on the test bars. The
        per-epoch ``train_loss`` / ``val_loss`` are logged to wandb and fed
        to the head's stopping rule; the loop never runs past
        ``config.epochs``. Torch is reseeded with ``config.random_seed``
        first, so a fit is reproducible on CPU.

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
            ``val_size`` or the purge leaves no timestamps to fit on.
        """
        config: DLConfig = self.config  # type: ignore[assignment]
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

        with Timer(f"{self.class_name}: to_array"):
            windows = self._windows(self.to_array(data, self.get_factor_names()))
            y = self.to_array(data, self.get_label_names())

        self.model = self._init_model(
            num_features=self.num_factors,
            num_labels=self.num_labels,
            hyperparameters=config.hyperparameters,
        ).to(self.device)
        optim = self._init_optim(self.model)  # type: ignore[arg-type]
        if optim is not None:
            self.optim = optim
        monitor = self.stopping.monitor()
        rng = np.random.default_rng(config.random_seed)

        epoch = 0
        for epoch in tqdm(range(config.epochs), desc=f"{self.class_name}_train"):
            train_loss = self._train_epoch(epoch, windows, y, train_bars, rng)
            val_loss = (
                self._val_loss(epoch, windows, y, val_bars) if len(val_bars) else None
            )
            if len(test_bars):
                self._test_epoch(epoch, windows, y, test_bars)
            if self._wandb_recorder is not None:
                logged = {"train_loss": train_loss}
                if val_loss is not None:
                    logged["val_loss"] = val_loss
                self._wandb_recorder.log(logged, step=epoch)
            if monitor.update(train_loss, val_loss, self.model):  # type: ignore[arg-type]
                logger.info(f"{self.class_name}: stopping after epoch {epoch}")
                break
        monitor.finish(self.model)  # type: ignore[arg-type]

        with Timer(f"{self.class_name}: evaluate"):
            metrics = self._evaluate(epoch, "train", windows, y, train_bars)
            if len(val_bars):
                metrics.update(self._evaluate(epoch, "val", windows, y, val_bars))
            if len(test_bars):
                metrics.update(self._evaluate(epoch, "test", windows, y, test_bars))

        self._save_model(
            Path(config.model_save_dir) / project_name / experiment_name / model_name
        )
        if self._wandb_recorder is not None:
            self._wandb_recorder.finish()
        self.optim = None
        return metrics

    def _predict(self, data: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Return ``[T, S, L]`` predictions for a ``[T, S, F]`` input.

        Bar ``t`` is predicted from the windows ending at ``t`` inside
        ``data`` itself, so the first N - 1 bars have zero rows for the
        missing history. A cell outside its bar's cross-section is NaN.

        Raises
        ------
        TypeError
            If ``data`` is neither a tensor nor an array.
        """
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        elif not isinstance(data, np.ndarray):
            raise TypeError(f"Unsupported data type: {type(data)}")
        windows = self._windows(data)
        return torch.from_numpy(self._predict_bars(windows, range(windows.num_times)))

    def _predict_panel_array(self, x: np.ndarray) -> np.ndarray:
        """Return ``predict(x)`` as a numpy ``[T, S, L]`` array."""
        return self.predict(x).numpy()  # type: ignore[union-attr]

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


class MLModel(BaseModel):
    """Numpy variant for tree models and other non-torch libraries.

    There is no epoch loop and no copy-based rollback. Training, early
    stopping and the choice of the best model are left to the library's own
    mechanism inside ``_fit_model``: boosting libraries decide early stopping
    per round with cached validation scores, and rolling back to the best
    round is a matter of keeping the first ``k`` trees, both of which an
    outer epoch loop would only make coarser and slower.

    A head implements four hooks: ``_init_model``, ``_preprocess``,
    ``_fit_model`` and ``_forward``. ``_loss``, ``_evaluate``,
    ``_resolved_hyperparameters`` and the inherited ``_compute_metrics`` have
    default implementations that may be overridden. Checkpoints are ``.joblib`` files
    written through ``MlBackend``; they are pickles, so only load files you
    trust.

    Examples
    --------
    A minimal head that predicts the first feature for every label::

        >>> class FirstFeatureHead(MLModel):
        ...     def _init_model(self, num_features, num_labels, hyperparameters):
        ...         return {"num_labels": num_labels}
        ...     def _preprocess(self, data):
        ...         return np.array(data, dtype=np.float64, copy=True)
        ...     def _fit_model(self, train_x, train_y, val_x, val_y):
        ...         pass
        ...     def _forward(self, x):
        ...         return np.repeat(x[..., :1], self.model["num_labels"], axis=-1)
        >>> head = FirstFeatureHead(MLConfig(
        ...     factors=[factor], labels=[label], model_save_dir="checkpoints",
        ...     factor_data_strategy="read", label_data_strategy="read",
        ...     train_start="2024-01-01", train_end="2024-01-30",
        ...     test_start="2024-01-31", test_end="2024-02-09",
        ... ))
        >>> head.collect().train().suffix
        '.joblib'
    """

    config_cls = MLConfig
    checkpoint_suffix = ".joblib"

    @abstractmethod
    def _init_model(
        self, num_features: int, num_labels: int, hyperparameters: dict
    ):
        """Prepare the model for the given shape; the result becomes ``self.model``.

        Tree libraries often build the real model only inside ``_fit_model``,
        in which case this may just resolve the hyperparameters and return
        None. ``load()`` does not call it: the checkpoint holds the whole
        model.
        """

    @abstractmethod
    def _preprocess(self, data: np.ndarray) -> np.ndarray:
        """Preprocess a ``[T, S, *]`` array and return a new one; never modify in place.

        Called once per training array (train and test, x and y) and once on
        the input at inference.
        """

    @abstractmethod
    def _fit_model(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        val_x: np.ndarray | None,
        val_y: np.ndarray | None,
    ) -> None:
        """Fit the model on ``[T, S, F]`` features and ``[T, S, L]`` labels.

        ``val_x`` and ``val_y`` are None when the validation segment is empty
        (``val_size == 0``). Early stopping and rollback to the best model are
        the hook's job, using the library's native mechanism and honouring
        ``config.early_stopping`` and ``config.early_stopping_patience``. On
        return ``self.model`` must be the model to save.
        """

    @abstractmethod
    def _forward(self, x: np.ndarray) -> np.ndarray:
        """Return ``[T, S, L]`` predictions for a preprocessed ``[T, S, F]`` input."""

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

    def _loss(self, y: np.ndarray, pred: np.ndarray) -> float:
        """Return the MSE over all labels at positions where every label is finite.

        Returns NaN when no position qualifies.
        """
        y = np.asarray(y, dtype=np.float64)
        pred = np.asarray(pred, dtype=np.float64)
        rows = np.isfinite(y).all(axis=-1)
        n = int(rows.sum())
        if n == 0:
            return float("nan")
        diff = pred[rows] - y[rows]
        return float(np.sum(diff * diff) / diff.size)

    def _evaluate(
        self, split: str, x: np.ndarray, y: np.ndarray
    ) -> dict[str, float]:
        """Evaluate one split and write the prefixed metrics to the wandb summary.

        Keys are ``{split}_loss`` and ``{split}_{mse,rmse,mae,r2,ic,rank_ic}``.
        They go to the run summary (final values, no step), so they do not
        interfere with per-round ``log(step=...)`` curves.
        """
        pred = self._forward(x)
        metrics = {f"{split}_loss": self._loss(y, pred)}
        for key, value in self._compute_metrics(y, pred).items():
            metrics[f"{split}_{key}"] = value
        if self._wandb_recorder is not None:
            self._wandb_recorder.summary.update(metrics)
        return metrics

    def _fit(
        self, project_name: str, experiment_name: str, model_name: str
    ) -> dict:
        """Split, fit once with ``_fit_model``, evaluate, save and finish the run.

        The validation segment is the trailing ``val_size`` share of the
        training window, and the purge of ``_fit_segments`` drops the last L
        bars before validation and before test, as in the torch variant.
        Empty splits skip
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
            unset, or ``val_size`` or the purge leaves no timestamps to fit
            on.
        """
        train_start, train_end, test_start, test_end = (
            self.config.train_start,
            self.config.train_end,
            self.config.test_start,
            self.config.test_end,
        )
        if not train_start or not train_end or not test_start or not test_end:
            raise ValueError(
                "Training and testing start and end dates must be specified."
            )

        self.model = self._init_model(
            num_features=self.num_factors,
            num_labels=self.num_labels,
            hyperparameters=self.config.hyperparameters,
        )
        resolved = self._resolved_hyperparameters()
        if resolved is not None and self._wandb_recorder is not None:
            # The run was opened in `_init_wandb`, before the hyperparameters
            # were resolved; record them now.
            self._wandb_recorder.config.update(
                {"resolved_hyperparameters": dict(resolved)},
                allow_val_change=True,
            )

        data = self.data_backend.get_xarray_dataset(["timestamp", "symbol"])
        train_data, val_data, test_data = self._fit_segments(data)
        factors = self.get_factor_names()
        labels = self.get_label_names()

        with Timer(f"{self.class_name}: to_array"):
            train_x, train_y, val_x, val_y, test_x, test_y = [
                self._preprocess(self.to_array(d, names))
                for d, names in [
                    (train_data, factors),
                    (train_data, labels),
                    (val_data, factors),
                    (val_data, labels),
                    (test_data, factors),
                    (test_data, labels),
                ]
            ]
        for d in [train_x, val_x, test_x]:
            self._assert_shape_match_x(d)
        for d in [train_y, val_y, test_y]:
            self._assert_shape_match_y(d)
        if val_x.shape[0] == 0:
            val_x = val_y = None

        with Timer(f"{self.class_name}: fit_model"):
            self._fit_model(train_x, train_y, val_x, val_y)

        with Timer(f"{self.class_name}: evaluate"):
            metrics = self._evaluate("train", train_x, train_y)
            if val_x is not None:
                metrics.update(self._evaluate("val", val_x, val_y))
            if test_x.shape[0] > 0:
                metrics.update(self._evaluate("test", test_x, test_y))

        self._save_model(
            Path(self.config.model_save_dir)
            / project_name
            / experiment_name
            / model_name,
        )

        if self._wandb_recorder is not None:
            self._wandb_recorder.finish()

        return metrics

    def _predict(self, data: torch.Tensor | np.ndarray) -> np.ndarray:
        """Preprocess ``data`` (tensors are converted to numpy) and run ``_forward``.

        Raises
        ------
        TypeError
            If ``data`` is neither a tensor nor an array.
        """
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Unsupported data type: {type(data)}")
        return self._forward(self._preprocess(data))

    def _predict_panel_array(self, x: np.ndarray) -> np.ndarray:
        """Return ``predict(x)`` as an array; ``_forward`` yields ``[T, S, L]``."""
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
