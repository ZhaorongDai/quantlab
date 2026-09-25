"""Model layer: the training lifecycle shared by every model head.

A *head* is one concrete predictive model, for example an MLP or an XGBoost
regressor. It reads *features* (factor values) and learns *labels* (the
targets to predict, typically forward returns). Both arrive as *panels*: an
``xarray.Dataset`` indexed by ``timestamp`` and ``symbol``, with one data
variable per feature or label.

The module defines a three-level class hierarchy. ``BaseModel`` holds
everything that does not depend on the training framework: config
validation, pushing the model's dates down to its factors and labels,
collecting their panels into one dataset, the public ``train`` /
``train_cv`` / ``load`` / ``predict`` / ``predict_panel`` methods, the
checkpoint directory layout with its ``config.json`` sidecar file, and the
fold boundaries of rolling cross-validation. ``DLModel`` is the PyTorch
variant: an epoch loop over ``DataLoader`` batches with early stopping and
``.pth`` checkpoints. ``MLModel`` is the numpy variant for tree models and
other libraries that do their own early stopping, with ``.joblib``
checkpoints. Concrete heads live in ``quantlab/dl_model`` and
``quantlab/ml_model``.
"""

import copy
import json
import random
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
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from quantlab.backend import XrBackend
from quantlab.enums.constant import Date
from quantlab.ml_model.backend import MlBackend
from quantlab.utils.atomic import write_json_atomically
from quantlab.utils.jsonable import to_jsonable
from quantlab.utils.metrics import regression_panel_metrics
from quantlab.utils.symbol_axis import sort_symbol_axis
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
    symbol_labeller : callable or None
        Optional ``(symbols, day) -> list[str]`` function that turns symbol
        labels into readable text for log messages. The backtester sets it.

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
        # Symbols of the training panel, recorded in the checkpoint sidecar.
        # None when no record exists. Elements keep the axis type (int or str).
        self._trained_symbols: list | None = None
        # The backtester checks the checkpoint before collecting data and
        # `load()` checks it again; remembering warnings logs each only once.
        self._emitted_load_warnings: set[str] = set()
        # Used only to make log messages readable. The model layer knows no
        # data vendors, so the caller (the backtester) supplies it.
        self.symbol_labeller = None

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
        """Install ``config`` and push its dates down to every factor and label.

        The type check runs before anything else so that a model handed the
        wrong config class fails before any factor or label object has been
        modified. Missing ``start_date`` / ``end_date`` fall back to the
        project-wide defaults, and ``config.name`` is set to the model's import
        path.

        Raises
        ------
        TypeError
            If ``config`` is not an instance of ``config_cls``.

        Examples
        --------
        >>> model.config = MLConfig(factors=[factor], labels=[label],
        ...                         model_save_dir="checkpoints",
        ...                         factor_data_strategy="read",
        ...                         label_data_strategy="read")
        >>> factor.config.start_date == model.config.start_date
        True
        """
        if not isinstance(config, self.config_cls):
            raise TypeError(
                f"{self.class_name} requires a {self.config_cls.__name__}, "
                f"got {type(config).__name__}"
            )
        self._config = config
        self._config.name = self.import_path

        if self._config.start_date is None:
            self._config.start_date = Date.START_DATE
        if self._config.end_date is None:
            self._config.end_date = Date.END_DATE

        self._reset_factors_config()
        self._reset_labels_config()

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

    def _reset_factors_config(self):
        """Copy the model's dates onto every factor and re-derive their datasets."""
        for factor in self._config.factors:
            factor.config.start_date = self._config.start_date
            factor.config.end_date = self._config.end_date

            factor._reset_dataset_config()

    def _reset_labels_config(self):
        """Copy the model's dates onto every label and re-derive their datasets."""
        for label in self._config.labels:
            label.config.start_date = self._config.start_date
            label.config.end_date = self._config.end_date

            label._reset_dataset_config()

    def _collect_all_labels(self) -> xr.Dataset:
        """Gather every label in ``config.labels`` into one sorted panel.

        Each label is computed (``cal``) or read from its store (``read``)
        according to ``config.label_data_strategy``.

        Raises
        ------
        ValueError
            If the strategy is neither ``"cal"`` nor ``"read"``.
        """
        all_ds = []
        for label in self.config.labels:
            match self.config.label_data_strategy:
                case "cal":
                    ds = label.cal().get_labels()
                case "read":
                    ds = label.read().get_labels()
                case _:
                    raise ValueError(
                        f"label_data_strategy {self.config.label_data_strategy} is not supported"
                    )
            all_ds.append(ds)
        data: xr.Dataset = xr.combine_by_coords(all_ds)  # type: ignore
        data = data.sortby(["timestamp", "symbol"])
        return data

    def _collect_all_features(self) -> xr.Dataset:
        """Gather every factor in ``config.factors`` into one panel.

        Each factor is computed (``cal``) or read from its store (``read``)
        according to ``config.factor_data_strategy``.

        Raises
        ------
        ValueError
            If the strategy is neither ``"cal"`` nor ``"read"``.
        """
        all_ds = []
        for factor in self.config.factors:
            match self.config.factor_data_strategy:
                case "cal":
                    ds = factor.cal().get_features()
                case "read":
                    ds = factor.read().get_features()
                case _:
                    raise ValueError(
                        f"data_strategy {self.config.factor_data_strategy} not supported"
                    )
            all_ds.append(ds)
        data: xr.Dataset = xr.combine_by_coords(all_ds)  # type: ignore
        return data

    def collect(
        self,
    ) -> Self:
        """Load features and labels into the model's data backend.

        The factor and label panels are merged on their shared
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
            d = xr.combine_by_coords([feature, label])
            d = d.sortby(["timestamp", "symbol"])
        self.data_backend.to_internal(d)  # type: ignore
        return self

    @staticmethod
    def _variable_names(obj) -> tuple[str, ...]:
        """Return the variables a factor or label object actually provides.

        This is ``obj.get_factor_names()``, i.e. ``config.factor_names``: a
        factor pinned to a subset computes only that subset, and ``Factor``
        fills an unset ``config.factor_names`` from ``_get_factor_names()``.
        Objects without ``get_factor_names`` (lightweight stand-ins) and the
        legacy ``["_all_"]`` placeholder fall back to ``_get_factor_names()``.

        Examples
        --------
        >>> alpha158.config.factor_names       # pinned to two features
        ('KMID', 'STD5')
        >>> BaseModel._variable_names(alpha158)
        ('KMID', 'STD5')
        """
        getter = getattr(obj, "get_factor_names", None)
        pinned = getter() if callable(getter) else None
        if pinned is not None and list(pinned) != ["_all_"]:
            return tuple(pinned)
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
        symbol axis, so that is the layout the head was trained on. The same
        symbols are kept on ``_trained_symbols`` for predictions made right
        after training.

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

        self._trained_symbols = symbols
        self._write_checkpoint(p)

    def load(self, p: Path | str) -> Self:
        """Restore the model from a checkpoint file.

        The file suffix is checked first, so a ``.pth`` file is never handed to
        joblib and a ``.joblib`` file never to ``torch.load``. The feature and
        label variables recorded in the sidecar ``config.json`` must then match
        this model's declared variables, name for name and in order; neither
        torch nor a tree library would notice a permuted or substituted input
        by itself. Finally the training symbols are read from the sidecar and
        the checkpoint is loaded.

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
        self._trained_symbols = self._read_trained_symbols(p)
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

    def _read_trained_symbols(self, p: Path) -> list | None:
        """Return the training symbols recorded beside checkpoint ``p``, or None.

        The values are returned as JSON decoded them (integers stay ``int``,
        strings stay ``str``) and are sorted with ``sort_symbol_axis``; only
        the membership of the record is authoritative, never its order.
        """
        saved = self._read_checkpoint_sidecar(p)
        record = saved.get(self.TRAINED_ON_KEY) if saved is not None else None
        symbols = record.get("symbols") if isinstance(record, dict) else None
        if not isinstance(symbols, list):
            return None
        return sort_symbol_axis(symbols)

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
        predictions for symbols that do not exist yet.

        The variant hook ``_align_prediction_symbols`` may restrict the symbol
        axis first (the torch variant aligns it to the training symbols), and
        ``_predict_panel_array`` performs the numeric prediction.

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

        # Sort again after the hook so the coordinates and the array layout
        # come from the same panel, whatever the hook returned.
        feats = self._align_prediction_symbols(
            features[factors].sortby(["timestamp", "symbol"])
        ).sortby(["timestamp", "symbol"])
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

    def _align_prediction_symbols(self, feats: xr.Dataset) -> xr.Dataset:
        """Symbol-axis hook of ``predict_panel``; the default returns ``feats`` as is.

        A numpy head predicts each ``(t, s)`` cell independently, so it does
        not care which symbols are present. Variants that encode symbol
        position (``DLModel``) override this to align the panel to the
        training symbols. It is a plain method rather than an abstract one so
        that the set of abstract methods of each variant stays unchanged.
        """
        return feats

    def _predict_panel_array(self, x: np.ndarray) -> np.ndarray:
        """Variant hook of ``predict_panel``: ``[T, S, F]`` array in, ``[T, S, L]`` out.

        A plain method rather than an abstract one, for the same reason as
        ``_align_prediction_symbols``.
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
        checkpoint. Returning the path lets a caller record exactly which
        model was trained and reload it later instead of retraining.

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
        """
        project_name = self._new_project_name()
        experiment_name = f"{self.class_name}_total"
        model_name = f"{experiment_name}{self.checkpoint_suffix}"
        self._init_wandb(
            project_name=project_name,
            experiment_name=experiment_name,
        )
        self._fit(
            project_name=project_name,
            experiment_name=experiment_name,
            model_name=model_name,
        )
        return (
            Path(self.config.model_save_dir) / project_name / experiment_name / model_name
        ).absolute()

    @staticmethod
    def _cv_folds(
        timestamps, train_periods: int, gap_periods: int
    ) -> list[dict]:
        """Compute the fold boundaries of a rolling walk-forward cross-validation.

        This is the only implementation of the fold arithmetic; both the
        sequential and the parallel branch of ``train_cv`` use it. With
        ``test_periods = train_periods // 5``, fold ``i`` trains on positions
        ``[i * test_periods, i * test_periods + train_periods)``, skips
        ``gap_periods`` positions, then tests on the next ``test_periods``
        positions. The number of folds is
        ``max(1, (len(timestamps) - train_periods - gap_periods) // test_periods)``;
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
        n_splits = max(
            1, (total_periods - train_periods - gap_periods) // test_periods
        )

        folds: list[dict] = []
        for i in range(n_splits):
            train_start_idx = i * test_periods
            train_end_idx = train_start_idx + train_periods
            test_start_idx = train_end_idx + gap_periods
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

    def _train_one_fold(self, fold: dict, project_name: str) -> dict:
        """Train one fold on this instance and return its result dict.

        The result is the fold dict plus ``experiment_name``, ``checkpoint``
        (the absolute path of the fold's checkpoint, since the manifest may be
        read from another working directory) and whatever ``test_*`` metrics
        ``_fit`` returned.
        """
        self.config.train_start = fold["train_start"]
        self.config.train_end = fold["train_end"]
        self.config.test_start = fold["test_start"]
        self.config.test_end = fold["test_end"]

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
            **fold,
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

    def _train_fold_with_config(self, fold: dict, project_name: str) -> dict:
        """Train one fold on a deep copy of this instance (parallel branch).

        Each fold gets its own copy so folds share no config dates, model or
        wandb run; the price is one copy of the panel per job.
        """
        return copy.deepcopy(self)._train_one_fold(fold, project_name)

    #: Name of the fold manifest ``train_cv`` writes into the trial directory.
    CV_FOLDS_FILENAME = "cv_folds.json"
    #: Format version written into the manifest. Readers reject versions they
    #: do not know, so bump this whenever the manifest structure changes.
    CV_FOLDS_FORMAT_VERSION = 1

    #: Keys every fold dict carries. ``test_start`` / ``test_end`` start with
    #: ``test_`` but are dates, not metrics, and are excluded from CV means.
    _CV_FOLD_KEYS = frozenset(
        {"fold", "train_start", "train_end", "test_start", "test_end"}
    )

    @staticmethod
    def _cv_mean_metrics(results: list[dict]) -> dict:
        """Average the ``test_*`` metrics over folds as ``cv_mean_{key}``.

        Only finite numeric values count; a metric with no finite value in
        any fold averages to NaN. ``cv_n_folds`` is added. Returns an empty
        dict when no fold carries a ``test_*`` metric (the torch variant's
        ``_fit`` returns none), in which case ``train_cv`` opens no summary run.
        """
        keys: list[str] = []
        for result in results:
            for key, value in result.items():
                if (
                    key.startswith("test_")
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
        gap_periods: int = 0,
        parallel: bool = False,
        njobs: int = -1,
    ) -> list[dict]:
        """Run a rolling walk-forward cross-validation and return per-fold results.

        Walk-forward cross-validation trains on a window of past data and
        tests on the period right after it, then slides both forward, so a
        test period never precedes its training data. Folds are laid out by ``_cv_folds`` over the timestamps between
        ``config.start_date`` and ``config.end_date``. Every fold trains on
        its own dates, gets its own wandb run and its own checkpoint directory
        ``{class}_cv_fold_{i}/`` inside one trial directory. The mean of the
        folds' ``test_*`` metrics is written to the summary of a separate
        ``{class}_cv_summary`` run.

        Before returning, the manifest ``cv_folds.json`` is written atomically
        into the trial directory as ``{"format_version": 1, "folds": [...]}``,
        where ``folds`` is the JSON form of the returned list (NaN and inf
        become null). Backtesters replay a CV run from that file.

        Parameters
        ----------
        train_periods : int
            Number of timestamps in each training segment. The test
            segment is one fifth of it.
        gap_periods : int, default 0
            Number of timestamps left out between a training segment and
            its test segment, so labels that look ahead cannot leak into
            the test.
        parallel : bool, default False
            Train the folds concurrently, each on a deep copy of this
            model, using a thread pool.
        njobs : int, default -1
            Number of threads for the parallel branch; ``-1`` uses all
            cores.

        Returns
        -------
        list[dict]
            One dict per fold: the fold boundaries, ``experiment_name``, the
            absolute ``checkpoint`` path and the fold's ``test_*`` metrics.

        Raises
        ------
        ValueError
            If no timestamps fall inside the config's date range.

        Examples
        --------
        >>> results = model.train_cv(train_periods=20, gap_periods=2)
        >>> len(results)
        4
        >>> results[0]["fold"], results[0]["checkpoint"].endswith("fold_0.joblib")
        (0, True)
        """
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
            f"Starting CV from {start_date} to {end_date} with {train_periods} training periods and {gap_periods} periods gap"
        )

        folds = self._cv_folds(timestamps, train_periods, gap_periods)

        logger.info(f"Total {len(folds)} folds will be created")
        for fold in folds:
            logger.info(
                f"Fold {fold['fold']}: Train [{fold['train_start']} to {fold['train_end']}], Test [{fold['test_start']} to {fold['test_end']}]"
            )

        if parallel:
            logger.info(f"Starting parallel training of {len(folds)} folds")
            results = list(
                Parallel(n_jobs=njobs, backend="threading")(
                    delayed(self._train_fold_with_config)(fold, project_name)
                    for fold in folds
                )
            )
        else:
            results = [
                self._train_one_fold(fold, project_name) for fold in folds
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
        experiment_name / model_name``. Returns the test metrics as a dict
        whose keys start with ``test_`` (``train_cv`` averages them), or None
        when the variant produces no metrics.
        """

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
    """Torch variant: an epoch loop over ``DataLoader`` batches.

    This layer owns the device, tensor conversion (``to_tensor``), the epoch
    loop with per-epoch early stopping and rollback to the best ``state_dict``,
    ``.pth`` checkpoints and the optional refit optimizer. A head implements
    five tensor hooks: ``_init_model``, ``_train_one_batch``,
    ``_val_one_batch``, ``_test_one_batch`` and ``_preprocess``, and in
    practice also ``_init_optim``. Inputs are ``[num_times, num_symbols,
    num_features]`` tensors, so a head sees every symbol of a bar at once and
    may encode symbol position; ``predict_panel`` therefore aligns the symbol
    axis to the training symbols before predicting.

    Early stopping is decided at epoch boundaries and rolls back to the best
    epoch's weights: a network has no cheaply sliceable structure like the
    first ``k`` trees of a boosted model, so the weights are snapshotted
    instead.

    Examples
    --------
    A minimal head; the base class supplies everything else::

        >>> class LinearHead(DLModel):
        ...     def _init_model(self, num_symbols, num_features, num_labels,
        ...                     hyperparameters):
        ...         return nn.Linear(num_features, num_labels)
        ...     def _init_optim(self, model):
        ...         return torch.optim.SGD(model.parameters(), lr=1e-3)
        ...     def _preprocess(self, data):
        ...         return torch.nan_to_num(data, nan=0.0)
        ...     def _train_one_batch(self, epoch, x, y):
        ...         self.optim.zero_grad()
        ...         loss = nn.functional.mse_loss(self.model(x), y)
        ...         loss.backward()
        ...         self.optim.step()
        ...         return loss.detach()
        ...     def _val_one_batch(self, epoch, x, y):
        ...         return nn.functional.mse_loss(self.model(x), y)
        ...     def _test_one_batch(self, epoch, x, y):
        ...         return nn.functional.mse_loss(self.model(x), y)
        >>> head = LinearHead(DLConfig(
        ...     factors=[factor], labels=[label], model_save_dir="checkpoints",
        ...     factor_data_strategy="read", label_data_strategy="read",
        ...     epochs=2, batch_size=8, num_workers=0,
        ...     train_start="2024-01-01", train_end="2024-01-30",
        ...     test_start="2024-01-31", test_end="2024-02-09",
        ... ))
        >>> head.collect().train().suffix
        '.pth'
    """

    config_cls = DLConfig
    checkpoint_suffix = ".pth"

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
        cpu
        """
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _to_default_float(tensor: torch.Tensor) -> torch.Tensor:
        """Cast a floating tensor to torch's default float dtype; others pass through.

        ``torch.from_numpy`` keeps numpy's dtype, and the data paths deliver
        float64 while model parameters are float32; without this cast the
        first linear layer raises a dtype mismatch.
        """
        default_dtype = torch.get_default_dtype()
        if tensor.is_floating_point() and tensor.dtype != default_dtype:
            tensor = tensor.to(default_dtype)
        return tensor

    def to_tensor(self, data: xr.Dataset, variables: list[str]) -> torch.Tensor:
        """Convert a panel to a ``[num_times, num_symbols, len(variables)]`` tensor.

        A thin wrapper over ``BaseModel.to_array``: the variable order comes
        from there, and this method only converts to a tensor in the default
        float dtype.

        Examples
        --------
        >>> t = head.to_tensor(panel, ["f_a", "f_b"])
        >>> t.shape, t.dtype
        (torch.Size([40, 3, 2]), torch.float32)
        """
        return self._to_default_float(
            torch.from_numpy(self.to_array(data, variables))
        )

    def _predict(self, data: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Run the network in eval mode under ``no_grad`` and return its output.

        Numpy input is converted to a default-float tensor. The model is left
        in eval mode afterwards; the training loop switches it back itself.

        Raises
        ------
        TypeError
            If ``data`` is neither a tensor nor an array.
        """
        if isinstance(data, np.ndarray):
            data = self._to_default_float(torch.from_numpy(data))
        elif not isinstance(data, torch.Tensor):
            raise TypeError(f"Unsupported data type: {type(data)}")
        # A freshly built nn.Module is in training mode: without eval() any
        # dropout stays active and the same input gives a different answer on
        # every call, and without no_grad() the graph is kept alive for nothing.
        self.model.eval()  # type: ignore[union-attr]
        with torch.no_grad():
            data = data.to(self.device)
            data = self._preprocess(data)
            return self.model(data)  # type: ignore[misc]

    def _predict_panel_array(self, x: np.ndarray) -> np.ndarray:
        """Convert the network's tensor output to a numpy ``[T, S, L]`` array.

        A head whose ``forward`` returns a tuple or list must override this
        hook and map its output onto one channel per label.

        Raises
        ------
        TypeError
            If the prediction is not a single tensor.
        """
        raw = self.predict(x)
        if isinstance(raw, torch.Tensor):
            return raw.detach().cpu().numpy()
        if isinstance(raw, (tuple, list)):
            raise TypeError(
                f"{self.class_name}: forward returns a {type(raw).__name__}, not "
                f"a [T, S, L] tensor; the head must override "
                f"_predict_panel_array to map it onto one channel per label"
            )
        raise TypeError(
            f"{self.class_name}: unsupported prediction type "
            f"{type(raw).__name__}; expected a torch.Tensor"
        )

    def _init_model_and_optim(self):
        """Build the network from the collected panel's shape and its optimizer.

        The network is moved to ``device``. ``_init_optim`` may return None
        to signal that the head updates parameters inside its own training
        hook; ``self.optim`` is then left untouched.
        """
        self.model = self._init_model(
            num_symbols=self.num_symbols,
            num_features=self.num_factors,
            num_labels=self.num_labels,
            hyperparameters=self.config.hyperparameters,
        )
        self.model = self.model.to(self.device)  # type: ignore
        optim = self._init_optim(self.model)  # type: ignore
        if optim is not None:
            self.optim = optim

    def _fit(
        self,
        project_name: str,
        experiment_name: str,
        model_name: str,
    ):
        """Train with the epoch loop, save the checkpoint and finish the wandb run.

        The training window is split by time: the first ``1 - val_size``
        share of its timestamps trains, the rest validates. Each epoch trains
        over shuffled batches, then evaluates the validation and test loaders
        without gradients. With ``config.early_stopping`` on, the
        sample-weighted mean of ``_val_one_batch`` is the epoch's validation
        loss; the weights of the best epoch are snapshotted and restored at
        the end, and training stops after ``early_stopping_patience`` epochs
        without improvement. Returns None: this variant reports no metrics.

        Raises
        ------
        ValueError
            If any of the four ``train_*`` / ``test_*`` dates is
            unset.
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

        self._init_model_and_optim()

        data = self.data_backend.get_xarray_dataset(["timestamp", "symbol"])
        train_data = data.sel(timestamp=slice(train_start, train_end))
        test_data = data.sel(timestamp=slice(test_start, test_end))
        factors = self.get_factor_names()
        labels = self.get_label_names()
        train_x = train_data[factors]
        train_y = train_data[labels]
        test_x = test_data[factors]
        test_y = test_data[labels]

        datas = [
            self.to_tensor(d, names)
            for d, names in [
                (train_x, factors),
                (train_y, labels),
                (test_x, factors),
                (test_y, labels),
            ]
        ]
        datas = [self._preprocess(d) for d in datas]

        train_x_t_all, train_y_t_all, test_x_t, test_y_t = datas
        for d in [train_x_t_all, test_x_t]:
            self._assert_shape_match_x(d)
        for d in [train_y_t_all, test_y_t]:
            self._assert_shape_match_y(d)

        train_split = int(train_x_t_all.shape[0] * (1 - self.config.val_size))
        train_x_t = train_x_t_all[:train_split]
        train_y_t = train_y_t_all[:train_split]
        # `train_split:` rather than `train_split + 1:`, so that no row falls
        # between the two splits.
        val_x_t = train_x_t_all[train_split:]
        val_y_t = train_y_t_all[train_split:]

        train_loader = DataLoader(
            TensorDataset(train_x_t, train_y_t),
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.config.num_workers,
        )

        val_test_loaders = [
            DataLoader(
                TensorDataset(*d),
                batch_size=self.config.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=self.config.num_workers,
            )
            for d in [
                (val_x_t, val_y_t),
                (test_x_t, test_y_t),
            ]
        ]
        val_loader, test_loader = val_test_loaders

        best_loss = float("inf")
        early_stopping = False
        patience = self.config.early_stopping_patience
        counter = 0
        best_state: dict[str, torch.Tensor] | None = None

        for epoch in tqdm(
            range(self.config.epochs),
            desc=f"{self.class_name}_train",
        ):
            self.model.train()  # type: ignore
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)
                self._train_one_batch(epoch, x_batch, y_batch)

            self.model.eval()  # type: ignore
            with torch.no_grad():
                val_loss_sum = 0.0
                val_sample_count = 0
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.to(self.device, non_blocking=True)
                    y_batch = y_batch.to(self.device, non_blocking=True)
                    val_loss = self._val_one_batch(epoch, x_batch, y_batch)

                    batch_samples = int(x_batch.shape[0])
                    val_loss_sum += float(val_loss) * batch_samples
                    val_sample_count += batch_samples

                if self.config.early_stopping and val_sample_count > 0:
                    epoch_val_loss = val_loss_sum / val_sample_count
                    if epoch_val_loss < best_loss:
                        best_loss = epoch_val_loss
                        counter = 0
                        best_state = {
                            k: v.detach().cpu().clone()
                            for k, v in self.model.state_dict().items()  # type: ignore[union-attr]
                        }
                    else:
                        counter += 1
                        if counter >= patience:
                            logger.info(f"Early stopping at epoch {epoch}")
                            early_stopping = True

                for x_batch, y_batch in test_loader:
                    x_batch = x_batch.to(self.device, non_blocking=True)
                    y_batch = y_batch.to(self.device, non_blocking=True)
                    self._test_one_batch(epoch, x_batch, y_batch)

                if early_stopping:
                    break

        if self.config.early_stopping and best_state is not None:
            self.model.load_state_dict(best_state)  # type: ignore[union-attr]

        self._save_model(
            Path(self.config.model_save_dir)
            / project_name
            / experiment_name
            / model_name,
        )

        if self._wandb_recorder:
            self._wandb_recorder.finish()

        self.optim = None

    def _align_prediction_symbols(self, feats: xr.Dataset) -> xr.Dataset:
        """Restrict ``feats`` to the sorted training symbols.

        A torch head encodes symbol position (an MLP flattens each bar to
        ``[S * F]``, a recurrent head steps along the symbol axis), so a panel
        with a different symbol set would silently misplace every prediction.
        When training symbols are known: a panel missing any of them raises
        ``ValueError``; symbols the model never saw are dropped with a warning
        and get no prediction; the result is ``feats.sel(symbol=sorted
        training symbols)``, the layout ``to_array`` produced during training.
        Only the membership of the record matters, never its order. Without a
        training record ``feats`` is returned as is.

        The symbol labels are compared in the panel's own spelling, and the
        label type is checked before membership (see
        ``_assert_symbol_types_match``).

        Raises
        ------
        ValueError
            If the panel lacks training symbols, or the label
            types of the record and the panel disagree.
        """
        trained = self._trained_symbols
        if trained is None:
            return feats
        present = list(feats.symbol.values.tolist())
        self._assert_symbol_types_match(trained, present, feats)
        present_set = set(present)
        missing = [symbol for symbol in trained if symbol not in present_set]
        as_of = self._panel_as_of(feats)
        if missing:
            shown = self._spell(missing[:20], as_of)
            raise ValueError(
                f"{self.class_name}.predict_panel: the feature panel lacks "
                f"{len(missing)} of the {len(trained)} symbols this model was "
                f"trained on: {shown}{' ...' if len(missing) > 20 else ''}. "
                f"A DL head encodes symbol position, so it cannot predict "
                f"without them"
            )
        trained_set = set(trained)
        extra = sort_symbol_axis(
            symbol for symbol in present_set if symbol not in trained_set
        )
        if extra:
            shown = self._spell(extra[:20], as_of)
            logger.warning(
                f"{self.class_name}.predict_panel: dropping {len(extra)} symbol(s) "
                f"the model was not trained on, which get no prediction: "
                f"{shown}{' ...' if len(extra) > 20 else ''}"
            )
        return feats.sel(symbol=sort_symbol_axis(trained))

    @staticmethod
    def _panel_as_of(feats: xr.Dataset):
        """Return the date of the panel's last bar, used to render symbol names.

        Symbol names change over time, so a lookup needs an as-of date; the
        end of the window is the spelling a reader of the message has in
        front of them. Returns None when the panel has no time axis.
        """
        stamps = feats["timestamp"].values if "timestamp" in feats.coords else []
        if len(stamps) == 0:
            return None
        return pd.Timestamp(stamps[-1]).date()

    def _spell(self, symbols: list, as_of) -> list[str]:
        """Render symbol labels as text for a log message.

        Without a ``symbol_labeller`` (or without an as-of date) the labels'
        own spelling is returned. The labeller is expected never to raise: a
        log message must not turn a warning into a crash.
        """
        if self.symbol_labeller is None or as_of is None:
            return [str(symbol) for symbol in symbols]
        return self.symbol_labeller(symbols, as_of)

    @staticmethod
    def _symbol_type_name(symbol) -> str:
        """Return a symbol label's kind: ``"int"`` for any integer, else its type name.

        ``numpy.int64`` and ``int`` are interchangeable for ``.sel()``, so
        they are reported as one kind rather than as a spurious mismatch.
        """
        if isinstance(symbol, bool):
            return type(symbol).__name__
        if isinstance(symbol, (int, np.integer)):
            return "int"
        return type(symbol).__name__

    def _assert_symbol_types_match(
        self, trained: list, present: list, feats: xr.Dataset
    ) -> None:
        """Refuse to align when the record and the panel use different label types.

        This runs before the membership check because that check cannot see
        the problem: stringified, every label is "found", and the failure only
        surfaces on the closing ``.sel()`` as a misleading ``KeyError`` about
        a missing index entry. Neither side is coerced: a string record
        coerced onto an integer axis could match the wrong columns silently.

        Raises
        ------
        ValueError
            If the two sides disagree on the label type.
        """
        if not trained or not present:
            return
        trained_kinds = {self._symbol_type_name(symbol) for symbol in trained}
        present_kinds = {self._symbol_type_name(symbol) for symbol in present}
        if trained_kinds == present_kinds:
            return
        raise ValueError(
            f"{self.class_name}._align_prediction_symbols: refusing to align "
            f"the feature panel onto this checkpoint's training record, "
            f"because the two disagree on the type of a symbol label. The "
            f"checkpoint records {sorted(trained_kinds)} (e.g. "
            f"{trained[0]!r}); the panel's 'symbol' coordinate has dtype "
            f"{feats.symbol.dtype!r} and holds {sorted(present_kinds)} "
            f"(e.g. {present[0]!r}). Neither side is converted to the other: "
            f"a record of tickers converted onto an axis of PERMNOs (CRSP's "
            f"permanent integer security ids) could silently match the wrong "
            f"columns. Either retrain on the current panel, or rewrite the "
            f"checkpoint's '{self.TRAINED_ON_KEY}.symbols' in the panel's own "
            f"spelling."
        )

    def _write_checkpoint(self, path: Path) -> None:
        """Save the network's ``state_dict`` to ``path`` with ``torch.save``."""
        torch.save(self.model.state_dict(), path)  # type: ignore[union-attr]

    def _read_checkpoint(self, path: Path) -> None:
        """Rebuild the network and load the ``state_dict`` stored at ``path``.

        The network's symbol count comes from the training record when one
        exists, so a checkpoint can be loaded without collecting data first
        and a symbol mismatch is caught later by ``predict_panel``; without a
        record the current panel's symbol count is used.
        """
        num_symbols = (
            len(self._trained_symbols)
            if self._trained_symbols is not None
            else self.num_symbols
        )
        self.model = self._init_model(
            num_symbols=num_symbols,
            num_features=self.num_factors,
            num_labels=self.num_labels,
            hyperparameters=self.config.hyperparameters,
        ).to(self.device)
        self.model.load_state_dict(torch.load(path))  # type: ignore[union-attr]

    @abstractmethod
    def _init_model(
        self,
        num_symbols: int,
        num_features: int,
        num_labels: int,
        hyperparameters: dict,
    ):
        """Build and return the ``nn.Module`` for the given panel shape.

        The base class moves it to ``device``. ``hyperparameters`` is
        ``config.hyperparameters``, a free-form dict.
        """

    @abstractmethod
    def _test_one_batch(
        self,
        epoch: int,
        x: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate one test batch; called each epoch in eval mode without gradients.

        The return value is not used by the base class; heads typically log
        metrics here.
        """

    @abstractmethod
    def _train_one_batch(
        self,
        epoch: int,
        x: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        """Run one optimisation step on one batch and return its loss.

        The base class has already called ``model.train()`` and moved ``x``
        and ``y`` to ``device``; the hook performs ``zero_grad``, forward,
        loss, ``backward`` and ``step``. ``x`` is ``[batch, num_symbols,
        num_features]`` and ``y`` is ``[batch, num_symbols, num_labels]``.
        """

    @abstractmethod
    def _val_one_batch(
        self,
        epoch: int,
        x: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        """Return the validation loss of one batch as a scalar tensor.

        Called in eval mode under ``no_grad``. The base class averages the
        returned values (weighted by batch size) into the epoch's validation
        loss, which drives early stopping and the best-epoch snapshot, so the
        result must be convertible with ``float()``.
        """

    @abstractmethod
    def _preprocess(self, data: torch.Tensor) -> torch.Tensor:
        """Preprocess a ``[T, S, *]`` tensor; shared by training and inference.

        Using one hook for both keeps the input distribution identical at
        training and prediction time. Heads typically replace NaN with zero.
        """

    def _init_optim(self, model: torch.nn.Module):
        """Return the optimizer for ``model``, or None if the head updates itself.

        The default raises ``NotImplementedError`` and ``_init_model_and_optim``
        calls it unguarded, so every head that trains must override it.
        """
        raise NotImplementedError

    def _get_refit_optim(self) -> torch.optim.Optimizer:
        """Return a cached AdamW optimizer with ``config.lr_refit`` for online refits.

        The optimizer is rebuilt only when ``self.model`` is replaced or
        ``lr_refit`` changes, so its state survives across refit calls.
        """
        key = (self.model, self.config.lr_refit)
        cached = getattr(self, "_refit_optim_cache", None)
        if (
            cached is not None
            and cached[0][0] is key[0]
            and cached[0][1] == key[1]
        ):
            return cached[1]

        optim = torch.optim.AdamW(
            self.model.parameters(),  # type: ignore[union-attr]
            lr=self.config.lr_refit,
        )
        self._refit_optim_cache = (key, optim)
        return optim


class MLModel(BaseModel):
    """Numpy variant for tree models and other non-torch libraries.

    There is no epoch loop and no copy-based rollback. Training, early
    stopping and the choice of the best model are left to the library's own
    mechanism inside ``_fit_model``: boosting libraries decide early stopping
    per round with cached validation scores, and rolling back to the best
    round is a matter of keeping the first ``k`` trees, both of which an
    outer epoch loop would only make coarser and slower.

    A head implements four hooks: ``_init_model``, ``_preprocess``,
    ``_fit_model`` and ``_forward``. ``_loss``, ``_compute_metrics``,
    ``_evaluate`` and ``_resolved_hyperparameters`` have default
    implementations that may be overridden. Checkpoints are ``.joblib`` files
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

    def _compute_metrics(self, y: np.ndarray, pred: np.ndarray) -> dict:
        """Return ``regression_panel_metrics`` for the primary label (index 0)."""
        return regression_panel_metrics(pred[..., 0], y[..., 0])

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
        training window, as in the torch variant. Empty splits skip
        evaluation: no validation segment means no ``val_*`` metrics, and an
        empty test segment returns ``{}``.

        Returns
        -------
        dict
            The ``test_*`` metrics dict.

        Raises
        ------
        ValueError
            If any of the four ``train_*`` / ``test_*`` dates is
            unset, or ``val_size`` leaves no timestamps to fit on.
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
        train_data = data.sel(timestamp=slice(train_start, train_end))
        test_data = data.sel(timestamp=slice(test_start, test_end))
        factors = self.get_factor_names()
        labels = self.get_label_names()

        with Timer(f"{self.class_name}: to_array"):
            train_x_all, train_y_all, test_x, test_y = [
                self._preprocess(self.to_array(d, names))
                for d, names in [
                    (train_data, factors),
                    (train_data, labels),
                    (test_data, factors),
                    (test_data, labels),
                ]
            ]
        for d in [train_x_all, test_x]:
            self._assert_shape_match_x(d)
        for d in [train_y_all, test_y]:
            self._assert_shape_match_y(d)

        n_train_times = train_x_all.shape[0]
        train_split = int(n_train_times * (1 - self.config.val_size))
        if train_split == 0:
            raise ValueError(
                f"Empty training segment: val_size={self.config.val_size} "
                f"leaves 0 of {n_train_times} training timestamps for fitting."
            )
        train_x = train_x_all[:train_split]
        train_y = train_y_all[:train_split]
        if train_split < n_train_times:
            val_x = train_x_all[train_split:]
            val_y = train_y_all[train_split:]
        else:
            val_x = val_y = None

        with Timer(f"{self.class_name}: fit_model"):
            self._fit_model(train_x, train_y, val_x, val_y)

        with Timer(f"{self.class_name}: evaluate"):
            self._evaluate("train", train_x, train_y)
            if val_x is not None:
                self._evaluate("val", val_x, val_y)
            test_metrics = (
                self._evaluate("test", test_x, test_y) if test_x.shape[0] > 0 else {}
            )

        self._save_model(
            Path(self.config.model_save_dir)
            / project_name
            / experiment_name
            / model_name,
        )

        if self._wandb_recorder is not None:
            self._wandb_recorder.finish()

        return test_metrics

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
