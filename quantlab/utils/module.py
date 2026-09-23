"""Rebuild datasets, factors, models and backtesters from their serialised configs.

Every configurable object records the class that built it as a dotted import
path in ``config.name`` (the ``import_path`` property of the base classes),
and that config is written as JSON next to model checkpoints and backtest
runs. The loaders here read such a dict back, import the named class, rebuild
any nested objects (a factor's dataset, a model's factors and labels, a
backtester's price dataset and model) and construct the object with the config
class the class itself declares through ``config_cls``.

Example:
    >>> import json
    >>> from quantlab.utils.module import load_backtester_from_config
    >>> config = json.load(open("runs/2024-06-01/config.json"))
    >>> backtester = load_backtester_from_config(config)
    >>> result = backtester.run()
"""

import copy
import importlib


def get_cls_from_path(path: str):
    """Import ``path`` (``"pkg.module.ClassName"``) and return the class.

    Args:
        path: A dotted path whose last segment is the attribute to fetch.

    Returns:
        The attribute named by the final segment, normally a class.

    Raises:
        ModuleNotFoundError: If the module part cannot be imported. Configs
            written under a previous package layout are not remapped.
        AttributeError: If the module has no such attribute.
    """
    module_path, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _config_cls_of(cls) -> type:
    """Return the config class ``cls`` declares in ``config_cls``.

    Guessing a config class is how a config dict of one factor backend used to
    be rebuilt as another's, and it would also let any importable callable be
    instantiated with a config dict, so a class without ``config_cls`` is
    refused.

    Raises:
        TypeError: If ``cls`` has no ``config_cls`` attribute, or it is not a
            type.
    """
    config_cls = getattr(cls, "config_cls", None)
    if not isinstance(config_cls, type):
        raise TypeError(
            f"{cls.__qualname__} declares no config_cls, so it cannot be "
            f"rebuilt from a config dict."
        )
    return config_cls


def load_dataset_from_config(config: dict):
    """Rebuild a dataset from its config dict.

    The class named in ``config["name"]`` is imported and constructed with its
    own declared config class. The input dict is deep-copied first and is
    returned to the caller unchanged.

    Args:
        config: The dict a dataset's ``config.to_dict()`` produced.

    Returns:
        A dataset instance.
    """
    config = copy.deepcopy(config)
    cls = get_cls_from_path(config["name"])
    return cls(_config_cls_of(cls)(**config))


def load_factor_from_config(config: dict):
    """Rebuild a factor, and the dataset nested inside it, from a config dict.

    The class named in ``config["name"]`` is resolved first. If it declares a
    callable ``from_config``, that classmethod receives the whole dict and owns
    the rebuild; this is how a factor that wraps another factor (and therefore
    has no dataset of its own) rebuilds its inner factor recursively through
    this same function. Otherwise the nested ``dataset`` dict is replaced by a
    rebuilt dataset and the factor is constructed with its declared config
    class. The caller's dict is never modified.

    Args:
        config: The dict a factor's ``config.to_dict()`` produced, including a
            nested ``dataset`` dict unless the class provides ``from_config``.

    Returns:
        A factor instance.
    """
    config = copy.deepcopy(config)
    cls = get_cls_from_path(config["name"])

    from_config = getattr(cls, "from_config", None)
    if callable(from_config):
        return from_config(config)

    config["dataset"] = load_dataset_from_config(config["dataset"])
    return cls(_config_cls_of(cls)(**config))


def load_model_from_config(config: dict):
    """Rebuild a model, with its factors and labels, from a config dict.

    ``cls.config_cls`` selects the right config class for the model variant
    (``DLConfig`` for torch heads, ``MLConfig`` for tree heads). Two keys a
    checkpoint's ``config.json`` carries as training records rather than
    config fields, ``resolved_hyperparameters`` and ``trained_on``, are dropped
    before construction; any other unknown key still fails loudly. The caller's
    dict is never modified.

    Args:
        config: The dict written beside a checkpoint, with ``factors`` and
            ``labels`` as lists of factor config dicts.

    Returns:
        A model instance (untrained; call ``load`` to restore a checkpoint).
    """
    config = copy.deepcopy(config)
    # `resolved_hyperparameters` (what the library actually trained with) and
    # `trained_on` (factor/label names and training symbols) are records, not
    # config fields. Drop only those so any other unknown key still fails.
    config.pop("resolved_hyperparameters", None)
    config.pop("trained_on", None)
    config["factors"] = [load_factor_from_config(f) for f in config["factors"]]
    config["labels"] = [load_factor_from_config(l) for l in config["labels"]]
    cls = get_cls_from_path(config["name"])
    return cls(cls.config_cls(**config))


def load_backtester_from_config(config: dict):
    """Rebuild a backtester from the ``config.json`` a backtest run wrote.

    The price dataset, the model (with its factors, labels and checkpoint
    reference), an optional benchmark dataset and every scalar parameter are
    rebuilt, and the backtester is constructed with its declared config class.
    Calling ``run()`` or ``run_cv()`` on the result re-runs the stored
    backtest.

    Two keys are records rather than config fields. ``data_fingerprint``
    describes the data the original run read; it is removed and assigned to
    the rebuilt backtester's ``expected_fingerprint`` so the re-run can warn
    when its data differs. ``trained_checkpoint`` names the checkpoint a
    train-mode run produced; rebuilding such a config retrains, so to replay
    that exact model set ``model_mode="load"`` and ``checkpoint`` to the
    recorded path.

    Every field of the config class must be present in the dict. Missing keys
    are not filled from the current dataclass defaults, because a default that
    changed since the run would silently produce a different backtest.

    Args:
        config: The dict read from a run directory's ``config.json``.

    Returns:
        A backtester instance ready to run.

    Raises:
        TypeError: If ``config["name"]`` is not a ``BaseBacktester`` subclass.
            This is checked before any nested dataset or model is built.
        ValueError: If any config field other than ``name`` is missing.
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
            f"from the values the stored backtest ran with (D-25)"
        )

    config["price_dataset"] = load_dataset_from_config(config["price_dataset"])
    config["model"] = load_model_from_config(config["model"])
    benchmark = config.get("benchmark_dataset")
    config["benchmark_dataset"] = (
        None if benchmark is None else load_dataset_from_config(benchmark)
    )

    backtester = cls(_config_cls_of(cls)(**config))
    backtester.expected_fingerprint = expected
    return backtester
