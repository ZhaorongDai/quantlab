"""Rebuild datasets, factors, models and backtesters from their serialised configs.

Every configurable object records the class that built it as a dotted import
path in ``config.name`` (the ``import_path`` property of the base classes),
and that config is written as JSON next to model checkpoints and backtest
runs. The loaders here read such a dict back, import the named class, rebuild
any nested objects (a factor's dataset, a model's factors and labels, a
backtester's price dataset and model) and construct the object with the config
class the class itself declares through ``config_cls``. This is what makes a
stored run reproducible from its ``config.json`` alone.

Examples
--------
>>> import json
>>> from quantlab.utils.module import load_backtester_from_config
>>> config = json.load(open("runs/2024-06-01/config.json"))
>>> backtester = load_backtester_from_config(config)
>>> result = backtester.run()
"""

import copy
import importlib
import os
from pathlib import Path


def get_cls_from_path(path: str):
    """Import ``path`` (``"pkg.module.ClassName"``) and return the class.

    Parameters
    ----------
    path : str
        A dotted path whose last segment is the attribute to fetch.

    Returns
    -------
    type
        The attribute named by the final segment, normally a class.

    Raises
    ------
    ModuleNotFoundError
        If the module part cannot be imported. Configs written before a
        module was moved or renamed are not remapped to the new path.
    AttributeError
        If the module has no such attribute.

    Examples
    --------
    >>> get_cls_from_path("quantlab.dataset.stock.StockDataset")
    <class 'quantlab.dataset.stock.StockDataset'>
    """
    module_path, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _config_cls_of(cls) -> type:
    """Return the config class ``cls`` declares in ``config_cls``.

    The config class is never guessed. A guess could rebuild one factor
    backend's config dict as another backend's config, and it would let any
    importable callable be instantiated with a config dict. A class without
    ``config_cls`` is therefore refused.

    Raises
    ------
    TypeError
        If ``cls`` has no ``config_cls`` attribute, or it is not a type.
    """
    config_cls = getattr(cls, "config_cls", None)
    if not isinstance(config_cls, type):
        raise TypeError(
            f"{cls.__qualname__} declares no config_cls, so it cannot be "
            f"rebuilt from a config dict."
        )
    return config_cls


def load_dataset_from_config(
    config: dict, *, run_dir: "str | os.PathLike | None" = None
):
    """Rebuild a dataset from its config dict.

    The class named in ``config["name"]`` is imported, its
    ``resolve_run_config(config, run_dir)`` prepares the config (a
    ``FrameDataset`` resolves a store a backtest run directory recorded
    relative to itself; every other dataset uses its paths as written), and
    the class is constructed with its own declared config class. The input
    dict is deep-copied first and is returned to the caller unchanged.

    Parameters
    ----------
    config : dict
        The dict a dataset's ``config.to_dict()`` produced.
    run_dir : str or os.PathLike, optional
        The run directory the config was read from.

    Returns
    -------
    BaseDataset
        A dataset instance.

    Raises
    ------
    ValueError
        If the class's ``resolve_run_config`` refuses the config, such as a
        ``FrameDataset`` store named relative to a run directory without
        ``run_dir``.

    Examples
    --------
    With ``dataset`` any dataset built earlier:

    >>> rebuilt = load_dataset_from_config(dataset.get_config())
    >>> type(rebuilt) is type(dataset), rebuilt.config == dataset.config
    (True, True)
    """
    config = copy.deepcopy(config)
    # Configs saved before `catalog_path` was removed still carry the key.
    config.pop("catalog_path", None)
    cls = get_cls_from_path(config["name"])
    config_cls = _config_cls_of(cls)
    if "datasets" in config:  # a merged dataset nests its inputs' configs
        config["datasets"] = [
            load_dataset_from_config(d, run_dir=run_dir) for d in config["datasets"]
        ]
    config = cls.resolve_run_config(config, None if run_dir is None else Path(run_dir))
    return cls(config_cls(**config))


def load_factor_from_config(config: dict):
    """Rebuild a factor or label, and what is nested inside it, from a config dict.

    The class named in ``config["name"]`` is resolved first. A factor's
    nested ``dataset`` dict is replaced by a rebuilt dataset; a label's
    (``quantlab.label.forward.Forward``) nested ``factor`` dict is replaced
    by a rebuilt factor, recursively. A market-feature factor's
    (``quantlab.factor.predefined.market.MarketFeatures``) ``series`` dict of dataset
    config dicts is rebuilt into datasets as well. The object is then
    constructed with its declared config class. The caller's dict is never
    modified.

    Parameters
    ----------
    config : dict
        The dict a factor's or label's ``get_config()`` produced, including
        a nested ``dataset`` or ``factor`` dict.

    Returns
    -------
    Factor or Forward
        A factor or label instance.

    Examples
    --------
    With ``factor`` any factor built earlier (``get_config`` nests its
    dataset's config under ``"dataset"``):

    >>> rebuilt = load_factor_from_config(factor.get_config())
    >>> type(rebuilt) is type(factor)
    True
    >>> rebuilt.config.factor_names == factor.config.factor_names
    True
    """
    config = copy.deepcopy(config)
    cls = get_cls_from_path(config["name"])
    if "factor" in config:  # a label nests the factor it shifts
        config["factor"] = load_factor_from_config(config["factor"])
    else:
        config["dataset"] = load_dataset_from_config(config["dataset"])
    if "series" in config:  # a market-feature factor nests its index datasets
        config["series"] = {
            name: load_dataset_from_config(dataset)
            for name, dataset in config["series"].items()
        }
    return cls(_config_cls_of(cls)(**config))


def load_model_from_config(config: dict):
    """Rebuild a model, with its factors and labels, from a config dict.

    The class named by ``config["name"]`` is imported and its own
    ``from_config`` rebuilds the model, so a model class decides how its
    config is read back. ``BaseModel.from_config`` rebuilds the factors and
    labels and drops the two training records a checkpoint's
    ``config.json`` carries, ``resolved_hyperparameters`` and ``trained_on``;
    any other unknown key still raises ``TypeError`` from the config class.
    The caller's dict is never modified.

    Parameters
    ----------
    config : dict
        The dict written beside a checkpoint, with ``factors`` and
        ``labels`` as lists of factor config dicts.

    Returns
    -------
    BaseModel
        A model instance (untrained; call ``load`` to restore a checkpoint).

    Examples
    --------
    Given the ``config.json`` written beside a checkpoint:

    >>> import json
    >>> with open("/data/models/xgb/config.json") as f:
    ...     config = json.load(f)
    >>> model = load_model_from_config(config)
    >>> model = model.load("/data/models/xgb/best.joblib")
    """
    return get_cls_from_path(config["name"]).from_config(config)


def load_backtester_from_config(
    config: dict, *, run_dir: "str | os.PathLike | None" = None
):
    """Rebuild a backtester from the ``config.json`` a backtest run wrote.

    The price dataset, the model (through ``from_config`` of the class its
    config names, so any ``Predictor`` rebuilds itself; ``None`` for a
    ``run_weights()`` run without one), the portfolio construction rule of a
    cross-sectional config (likewise through its class's ``from_config``), an
    optional benchmark dataset and every scalar parameter are rebuilt, and the backtester is constructed with its declared config class.
    Calling ``run()`` or ``run_cv()`` on the result re-runs the stored
    backtest; a ``run_weights()`` run is replayed by passing ``run_weights``
    the weights it simulated, ``XrBackend().read(run_dir / "weights.zarr").data``.

    The datasets are rebuilt through ``load_dataset_from_config`` with
    ``run_dir``. A run whose price or benchmark dataset was a ``FrameDataset``
    (every ``quantlab.api.backtest`` run) holds that panel under the run
    directory's ``inputs/``, and its config names the store relative to the
    run directory, so the directory can be moved; such a config needs
    ``run_dir``.

    Two keys are records rather than config fields. ``data_fingerprint``
    describes the data the original run read (time range, axis sizes and a
    sha256 digest of the values); it is removed and assigned to the rebuilt backtester's
    ``expected_fingerprint`` so the re-run can warn when its data differs.
    ``trained_checkpoint`` names the checkpoint a train-mode run produced.
    Rebuilding such a config retrains the model, so to replay that exact
    model, set ``model_mode="load"`` and ``checkpoint`` to the recorded path.

    Every field of the config class must be present in the dict. Missing keys
    are not filled from the current dataclass defaults, because a default that
    changed since the run would silently produce a different backtest.

    Parameters
    ----------
    config : dict
        The dict read from a run directory's ``config.json``.
    run_dir : str or os.PathLike, optional
        The run directory ``config`` was read from. Required when the config
        names ``inputs/`` stores, which are resolved against it.

    Returns
    -------
    BaseBacktester
        A backtester instance ready to run.

    Raises
    ------
    TypeError
        If ``config["name"]`` is not a ``BaseBacktester`` subclass. This is
        checked before any nested dataset or model is built.
    ValueError
        If any config field other than ``name`` is missing, or the config
        names ``inputs/`` stores and ``run_dir`` is not given.

    Examples
    --------
    Given the run directory of an earlier backtest:

    >>> import json
    >>> with open("/data/backtests/2024-06-01/config.json") as f:
    ...     config = json.load(f)
    >>> backtester = load_backtester_from_config(config)
    >>> result = backtester.run()

    A ``quantlab.api.backtest`` run kept with ``output_dir``, rebuilt from its
    directory and replayed from its saved weights:

    >>> import json, tempfile
    >>> import numpy as np
    >>> import pandas as pd
    >>> import quantlab.api as qa
    >>> from quantlab.backend import XrBackend
    >>> from quantlab.utils.module import load_backtester_from_config
    >>> bars = pd.bdate_range("2024-01-01", periods=5)
    >>> prices = pd.DataFrame({
    ...     "timestamp": np.repeat(bars, 2), "symbol": ["AAA", "BBB"] * 5,
    ...     "open": np.linspace(10.0, 14.0, 10), "close": np.linspace(10.5, 14.5, 10),
    ... })
    >>> weights = pd.DataFrame({"timestamp": [bars[0]], "symbol": ["AAA"],
    ...                         "weight": [1.0]})
    >>> report = qa.backtest(prices, weights=weights, output_dir=tempfile.mkdtemp())
    >>> run_dir = report.raw.run_dir
    >>> config = json.loads((run_dir / "config.json").read_text())
    >>> config["price_dataset"]["zarr_file_path"]
    'inputs/price_dataset.zarr'
    >>> backtester = load_backtester_from_config(config, run_dir=run_dir)
    >>> again = backtester.run_weights(XrBackend().read(run_dir / "weights.zarr").data)
    >>> json.loads((again.run_dir / "metrics.json").read_text()) == json.loads(
    ...     (run_dir / "metrics.json").read_text())
    True
    >>> again.simulation.value.values.round(2).tolist()
    [1000000.0, 1044873.23, 1126424.31, 1207975.4, 1289526.48]
    >>> (again.simulation.value == report.raw.simulation.value).all().item()
    True
    """
    # Imported here so this module does not import the backtest layer at
    # import time.
    from quantlab.base.backtest import BaseBacktester

    config = copy.deepcopy(config)
    expected = config.pop("data_fingerprint", None)
    config.pop("trained_checkpoint", None)

    cls = get_cls_from_path(config["name"])
    if not (isinstance(cls, type) and issubclass(cls, BaseBacktester)):
        raise TypeError(
            f"{config['name']} is not a BaseBacktester subclass, so it cannot be "
            f"rebuilt as a backtester"
        )

    from dataclasses import fields

    missing = [
        field.name
        for field in fields(_config_cls_of(cls))
        if field.name != "name" and field.name not in config
    ]
    if missing:
        raise ValueError(
            f"{config['name']} config is missing field(s) {missing}; refusing to "
            f"fill them from the current dataclass defaults, which may differ "
            f"from the values the stored backtest ran with"
        )

    config["price_dataset"] = load_dataset_from_config(
        config["price_dataset"], run_dir=run_dir
    )
    # The model is rebuilt by its own class, so any predictor (a model, or an
    # ensemble of models) round-trips without a special case here.
    # A config for run_weights() carries no model.
    model = config.get("model")
    config["model"] = (
        None
        if model is None
        else get_cls_from_path(model["name"]).from_config(model)
    )
    # A cross-sectional config holds its portfolio construction rule, rebuilt
    # by the class its config names.
    constructor = config.get("constructor")
    if constructor is not None:
        config["constructor"] = get_cls_from_path(constructor["name"]).from_config(
            constructor
        )
    benchmark = config.get("benchmark_dataset")
    config["benchmark_dataset"] = (
        None
        if benchmark is None
        else load_dataset_from_config(benchmark, run_dir=run_dir)
    )

    backtester = cls(_config_cls_of(cls)(**config))
    backtester.expected_fingerprint = expected
    return backtester
