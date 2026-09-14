"""Reconstruct a dataset/factor/model object from its serialized config.

A config records the class to rebuild as a dotted path in `config.name`, taken
from the `import_path` property the four base classes expose. Those paths are
therefore part of the on-disk format, and the namespace migration changed all
of them -- see `get_cls_from_path` for the decision and its evidence.
"""

import importlib

from quantlab.base.config import DatasetConfig, FactorConfig


def get_cls_from_path(path: str):
    """Import `path` ("pkg.module.ClassName") and return the class.

    DECISION (quick task 260907-sm2): the namespace migration is a HARD BREAK
    for configs persisted before it, and no legacy alias table is provided.

    Every dotted path this resolves originates in the `import_path` property of
    `BaseDataset`, `Factor`, `BaseModel` or `Acquisition`, which builds it from
    the class's own module and qualified name. Each config setter assigns that
    string to `config.name`, and it is serialized into the JSON written beside
    a model checkpoint. Moving the twelve former top-level packages under one
    umbrella package changed every one of those module names, so a config
    written before the migration names a module that no longer exists and
    raises `ModuleNotFoundError` here.

    THE EVIDENCE for taking the break rather than mapping the old names. A
    search of the whole repository, excluding version-control and virtualenv
    directories, returns zero `.pth` files and zero `.joblib` files -- the two
    formats a trained model is persisted in. The only `config.json` in the tree
    belongs to the planning tooling and is not a model config. Neither
    directory such artifacts would live in exists, both are excluded from
    version control, and no file of either extension has ever been added in
    this repository's history, so none is recoverable from an earlier commit.
    The set of configs an alias table would rescue is empty, and the table
    would be permanent maintenance owed to that empty set.

    The mapping stays available: should a pre-migration artifact ever turn up,
    rewriting its stale prefix here is a strictly additive change.
    """
    module_path, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def load_dataset_from_config(config: dict):
    return get_cls_from_path(config["name"])(DatasetConfig(**config))


def load_factor_from_config(config: dict):
    config["dataset"] = load_dataset_from_config(config["dataset"])
    return get_cls_from_path(config["name"])(FactorConfig(**config))


def load_model_from_config(config: dict):
    """Rebuild a model, using the config class the model class declares.

    `cls.config_cls` is `DLConfig` for `DLModel` heads and `MLConfig` for
    `MLModel` heads. Hardcoding `DLConfig` here used to turn an ML checkpoint's
    config into a `DLConfig` silently; the `BaseModel` config setter now also
    rejects a mismatched config type with `TypeError`.
    """
    config["factors"] = [load_factor_from_config(f) for f in config["factors"]]
    config["labels"] = [load_factor_from_config(l) for l in config["labels"]]
    cls = get_cls_from_path(config["name"])
    return cls(cls.config_cls(**config))
