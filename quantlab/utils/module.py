import importlib

from quantlab.base.config import DatasetConfig, DLConfig, FactorConfig


def get_cls_from_path(path: str):
    module_path, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def load_dataset_from_config(config: dict):
    return get_cls_from_path(config["name"])(DatasetConfig(**config))


def load_factor_from_config(config: dict):
    config["dataset"] = load_dataset_from_config(config["dataset"])
    return get_cls_from_path(config["name"])(FactorConfig(**config))


def load_model_from_config(config: dict):
    config["factors"] = [load_factor_from_config(f) for f in config["factors"]]
    config["labels"] = [load_factor_from_config(l) for l in config["labels"]]
    return get_cls_from_path(config["name"])(DLConfig(**config))
