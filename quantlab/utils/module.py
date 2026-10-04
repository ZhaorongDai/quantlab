"""Rebuild datasets, factors, models and backtesters from their serialised configs.

Every configurable object records the class that built it as a dotted import
path in ``config.name`` (the ``import_path`` property of the base classes),
and that config is written as JSON next to model checkpoints and backtest
runs. The loaders here read such a dict back, import the named class, rebuild
any nested objects (a factor's dataset, a model's factors and labels, a
backtester's price dataset and model) and construct the object with the config
class the class itself declares through ``config_cls``. This is what makes a
stored run reproducible from its ``config.json`` alone. The dataset and factor
loaders are thin wrappers over the component rule,
``quantlab.base.component.rebuild``.

Examples
--------
>>> import json
>>> from quantlab.utils.module import load_backtester_from_config
>>> config = json.load(open("runs/2024-06-01/config.json"))
>>> backtester = load_backtester_from_config(config)
>>> result = backtester.run()
"""

import importlib
import os


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


def load_dataset_from_config(
    config: dict, *, run_dir: "str | os.PathLike | None" = None
):
    """Rebuild a dataset from its config dict.

    A thin wrapper over the component rule,
    ``quantlab.base.component.rebuild``: the class named in
    ``config["name"]`` rebuilds itself and every dataset nested in it, and
    a ``FrameDataset`` store recorded relative to a run directory is
    resolved against ``run_dir`` at any depth. The caller's dict is not
    modified.

    Parameters
    ----------
    config : dict
        The dict a dataset's ``get_config()`` produced.
    run_dir : str or os.PathLike, optional
        The run directory the config was read from.

    Returns
    -------
    BaseDataset
        A dataset instance.

    Raises
    ------
    ValueError
        If the config holds an unknown key, or names a ``FrameDataset`` store
        relative to a run directory without ``run_dir``.

    Examples
    --------
    With ``dataset`` any dataset built earlier:

    >>> rebuilt = load_dataset_from_config(dataset.get_config())
    >>> type(rebuilt) is type(dataset), rebuilt.config == dataset.config
    (True, True)
    """
    from quantlab.base.component import rebuild

    return rebuild(config, run_dir)


def load_factor_from_config(
    config: dict, *, run_dir: "str | os.PathLike | None" = None
):
    """Rebuild a factor or label, and what is nested inside it, from a config dict.

    A thin wrapper over the component rule,
    ``quantlab.base.component.rebuild``: the class named in
    ``config["name"]`` rebuilds the components its config class declares (a
    factor's dataset, a label's factor, a market-feature factor's series)
    and is constructed with its declared config class. The caller's dict is
    never modified.

    Parameters
    ----------
    config : dict
        The dict a factor's or label's ``get_config()`` produced, including
        a nested ``dataset`` or ``factor`` dict.
    run_dir : str or os.PathLike, optional
        The run directory the config was read from, passed to every nested
        dataset.

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
    from quantlab.base.component import rebuild

    return rebuild(config, run_dir)


def load_model_from_config(config: dict):
    """Rebuild a model, with its factors and labels, from a config dict.

    A thin wrapper over the component rule,
    ``quantlab.base.component.rebuild``: the class named by
    ``config["name"]`` rebuilds itself through its own ``from_config``, so
    any predictor (a model, an ensemble) is read back. An unknown key is
    refused with ``ValueError``. The caller's dict is never modified.

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
    from quantlab.base.component import rebuild

    return rebuild(config)


def load_backtester_from_config(
    config: dict, *, run_dir: "str | os.PathLike | None" = None
):
    """Rebuild a backtester from the ``config.json`` a backtest run wrote.

    The class named by ``config["name"]`` is checked to be a backtester and
    rebuilds itself through ``BaseBacktester.from_config``: the records
    (``market``, ``data_fingerprint``, ``trained_checkpoint``) are taken out,
    a missing config field is refused, and the component rule rebuilds the
    datasets, model, construction rule and tracker. Calling ``run()`` or
    ``run_cv()`` on the result re-runs the stored backtest; a
    ``run_weights()`` run is replayed by passing ``run_weights`` the weights
    it simulated, ``XrBackend().read(run_dir / "weights.zarr").data``.

    A run whose price or benchmark dataset was a ``FrameDataset`` (every
    ``quantlab.api.backtest`` run) holds that panel under the run
    directory's ``inputs/``, and its config names the store relative to the
    run directory, so the directory can be moved; such a config needs
    ``run_dir``.

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
        If any config field other than ``name`` is missing, a key is
        unknown, or the config names ``inputs/`` stores and ``run_dir`` is
        not given.

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

    cls = get_cls_from_path(config["name"])
    if not (isinstance(cls, type) and issubclass(cls, BaseBacktester)):
        raise TypeError(
            f"{config['name']} is not a BaseBacktester subclass, so it cannot be "
            f"rebuilt as a backtester"
        )
    return cls.from_config(config, run_dir=run_dir)
