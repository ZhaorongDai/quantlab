"""Reconstruct a dataset/factor/model/backtester object from its serialized config.

A config records the class to rebuild as a dotted path in `config.name`, taken
from the `import_path` property the base classes expose. Those paths are
therefore part of the on-disk format, and the namespace migration changed all
of them -- see `get_cls_from_path` for the decision and its evidence.
"""

import copy
import importlib


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


def _config_cls_of(cls) -> type:
    """Return the config class `cls` declares, or refuse it by name.

    A class without `config_cls` gives the loader no way to know which config
    to build, and guessing one is exactly how a `PolarsFactorConfig` dict used
    to become a `FactorConfig` (D-26). The refusal also keeps an arbitrary
    importable callable from being instantiated with a config dict.
    """
    config_cls = getattr(cls, "config_cls", None)
    if not isinstance(config_cls, type):
        raise TypeError(
            f"{cls.__qualname__} declares no config_cls, so it cannot be "
            f"rebuilt from a config dict."
        )
    return config_cls


def load_dataset_from_config(config: dict):
    """Rebuild a dataset with the config class its class declares (D-26).

    `MarketDataset` declares `DatasetConfig` and `IndexConstituentDataset`
    declares `ConstituentDatasetConfig`. The input is deep-copied first:
    callers reuse the dict they saved, so it must come back untouched
    (RESEARCH Pitfall 9).
    """
    config = copy.deepcopy(config)
    cls = get_cls_from_path(config["name"])
    return cls(_config_cls_of(cls)(**config))


def load_factor_from_config(config: dict):
    """Rebuild a factor, and its nested dataset, with their declared config classes (D-26).

    `FactorKunQuant` declares `FactorConfig` and `FactorPolars` declares
    `PolarsFactorConfig`. The nested `dataset` dict is replaced by a rebuilt
    dataset object on a deep copy, never on the caller's dict (RESEARCH
    Pitfall 9).
    """
    config = copy.deepcopy(config)
    config["dataset"] = load_dataset_from_config(config["dataset"])
    cls = get_cls_from_path(config["name"])
    return cls(_config_cls_of(cls)(**config))


def load_model_from_config(config: dict):
    """Rebuild a model, using the config class the model class declares.

    `cls.config_cls` is `DLConfig` for `DLModel` heads and `MLConfig` for
    `MLModel` heads. Hardcoding `DLConfig` here used to turn an ML checkpoint's
    config into a `DLConfig` silently; the `BaseModel` config setter now also
    rejects a mismatched config type with `TypeError`.

    The input is deep-copied first so the caller's dict never receives the
    rebuilt factor objects (RESEARCH Pitfall 9).
    """
    config = copy.deepcopy(config)
    # `resolved_hyperparameters` is a record written by `MLModel.get_config`
    # (what the library actually trained with), not a config field: drop it,
    # and only it, so any other unknown key still fails loudly below.
    config.pop("resolved_hyperparameters", None)
    # `trained_on` is the training record `BaseModel._save_model` writes into
    # a checkpoint's config.json (factor/label names and the training symbols,
    # code review WR-02). Also a record, not a config field.
    config.pop("trained_on", None)
    config["factors"] = [load_factor_from_config(f) for f in config["factors"]]
    config["labels"] = [load_factor_from_config(l) for l in config["labels"]]
    cls = get_cls_from_path(config["name"])
    return cls(cls.config_cls(**config))


def load_backtester_from_config(config: dict):
    """Rebuild a backtester from the `config.json` a backtest run wrote (D-25).

    The price dataset, the model (with its factors, labels and checkpoint
    reference), an optional benchmark dataset and every scalar parameter are
    rebuilt, and the backtester is constructed with the config class its class
    declares (D-26). Calling `run()` or `run_cv()` on the result re-runs the
    stored backtest.

    `data_fingerprint` is a record of what the original run read, not a config
    field. It is removed from the config and assigned to the rebuilt
    backtester's `expected_fingerprint`, so the re-run compares the data it
    reads against the stored run and warns on a mismatch (D-27).

    The class named in `config["name"]` must be a `BaseBacktester` subclass.
    That is checked before any nested config is built, so a tampered name
    cannot get a dataset or model constructed on its behalf. A config JSON is
    otherwise trusted local input, like a checkpoint: it names classes to import
    and paths to read.

    The input is deep-copied first and comes back untouched (RESEARCH
    Pitfall 9). `BaseBacktester` is imported inside the function: this module
    must not import the backtest layer at import time.
    """
    from quantlab.base.backtest import BaseBacktester

    config = copy.deepcopy(config)
    expected = config.pop("data_fingerprint", None)

    cls = get_cls_from_path(config["name"])
    if not (isinstance(cls, type) and issubclass(cls, BaseBacktester)):
        raise TypeError(
            f"{config['name']} is not a BaseBacktester subclass, so it cannot be "
            f"rebuilt as a backtester"
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
