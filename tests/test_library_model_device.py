"""Tests for the default training device of the library heads (issue #72).

What is locked, and what turns it red:

- with no ``device`` hyperparameter, ``RealMLPRegressor`` and
  ``XGBoostRegressor`` resolve ``"cuda"`` when a CUDA device is available
  and ``"cpu"`` otherwise, and never Apple MPS, even when MPS is available;
- an explicit ``device`` is handed to the library unchanged;
- the resolved device is part of ``resolved_hyperparameters``;
- the XGBoost probe needs a CUDA build of xgboost AND a visible CUDA device,
  asks the CUDA driver without importing torch, and never raises;
- ``XGBTDRegressor`` passes no ``device`` (pytabkit's XGBoost path does not
  forward one to xgboost).

CUDA availability is patched both ways; no test needs a GPU.
"""

import subprocess
import sys

import numpy as np
import pytest
import torch
import xarray as xr

from quantlab.base.config import ModelConfig
from quantlab.model.predefined._support import devices
from quantlab.model.predefined.realmlp import RealMLPRegressor
from quantlab.model.predefined.xgb import XGBoostRegressor
from quantlab.model.predefined.xgb_td import XGBTDRegressor
from tests.label_stubs import StubLabel

TIMES = np.datetime64("2024-01-01") + np.arange(40).astype("timedelta64[D]")
SYMBOLS = ["A", "B", "C"]


class _Panel:
    def __init__(self, name):
        rng = np.random.default_rng(len(name))
        self.ds = xr.Dataset(
            {name: (("timestamp", "symbol"), rng.standard_normal((40, 3)).astype("float32"))},
            coords={"timestamp": TIMES, "symbol": SYMBOLS},
        )

    def _get_factor_names(self):
        return list(self.ds.data_vars)

    def read(self, start, end):
        return self.ds.sel(timestamp=slice(start, end))

    def get_config(self):
        return {"factor_names": self._get_factor_names()}


def _head(cls, tmp_path, hyperparameters=None):
    label = StubLabel(_Panel("ret"))
    config = ModelConfig(
        factors=[_Panel("f")], labels=[label], model_save_dir=str(tmp_path),
        factor_data_strategy="read", label_data_strategy="read",
        train_start="2024-01-01", train_end="2024-01-30",
        test_start="2024-01-31", test_end="2024-02-09",
        hyperparameters=dict(hyperparameters or {}),
    )
    return cls(config)


@pytest.fixture
def cuda(monkeypatch):
    """Return a setter that makes CUDA look available or not, to both libraries."""

    def set_available(available: bool) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
        monkeypatch.setattr(devices, "xgboost_cuda_available", lambda: available)

    return set_available


@pytest.fixture
def recorders_off(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_SILENT", "true")


@pytest.fixture
def mps(monkeypatch):
    """Make Apple MPS look available."""
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)


# --------------------------------------------------------------------------
# RealMLP
# --------------------------------------------------------------------------


@pytest.mark.parametrize("available, expected", [(True, "cuda"), (False, "cpu")])
def test_realmlp_defaults_to_cuda_when_available_else_cpu(tmp_path, cuda, mps, available, expected):
    cuda(available)
    head = _head(RealMLPRegressor, tmp_path)
    estimator = head._init_model(1, 1, head.config.hyperparameters)

    assert estimator.get_params()["device"] == expected
    assert head._resolved_hyperparameters()["device"] == expected
    assert "device" not in head.config.hyperparameters


@pytest.mark.parametrize("given", ["cpu", "cuda:0", "mps"])
def test_realmlp_honours_an_explicit_device(tmp_path, cuda, given):
    cuda(True)
    head = _head(RealMLPRegressor, tmp_path, {"device": given})
    estimator = head._init_model(1, 1, head.config.hyperparameters)

    assert estimator.get_params()["device"] == given
    assert head._resolved_hyperparameters()["device"] == given


def test_realmlp_treats_device_none_as_unset(tmp_path, cuda, mps):
    """pytabkit's own ``device=None`` would pick MPS on a Mac; the head never lets it."""
    cuda(False)
    head = _head(RealMLPRegressor, tmp_path, {"device": None})
    estimator = head._init_model(1, 1, head.config.hyperparameters)

    assert estimator.get_params()["device"] == "cpu"


def test_realmlp_default_params_pin_no_device():
    assert "device" not in RealMLPRegressor.DEFAULT_PARAMS


# --------------------------------------------------------------------------
# XGBoost
# --------------------------------------------------------------------------


@pytest.mark.parametrize("available, expected", [(True, "cuda"), (False, "cpu")])
def test_xgboost_defaults_to_cuda_when_available_else_cpu(tmp_path, cuda, mps, available, expected):
    cuda(available)
    head = _head(XGBoostRegressor, tmp_path)
    head._init_model(1, 1, head.config.hyperparameters)

    assert head._params["device"] == expected
    assert head._resolved_hyperparameters()["device"] == expected


@pytest.mark.parametrize("given", ["cpu", "cuda:1"])
def test_xgboost_honours_an_explicit_device(tmp_path, cuda, given):
    cuda(True)
    head = _head(XGBoostRegressor, tmp_path, {"device": given})
    head._init_model(1, 1, head.config.hyperparameters)

    assert head._resolved_hyperparameters()["device"] == given


def test_a_trained_xgboost_booster_predicts_on_the_cpu(tmp_path, recorders_off):
    """Whatever it trained on (CUDA on a GPU host), the Booster predicts numpy rows on the CPU."""
    import json

    head = _head(XGBoostRegressor, tmp_path, {"num_boost_round": 2})
    head.collect().train()
    booster = json.loads(head.model.save_config())
    assert booster["learner"]["generic_param"]["device"] == "cpu"


def test_xgboost_default_params_pin_no_device():
    assert "device" not in XGBoostRegressor.DEFAULT_PARAMS


# --------------------------------------------------------------------------
# The xgboost CUDA probe
# --------------------------------------------------------------------------


@pytest.fixture
def fresh_probe():
    devices.xgboost_cuda_available.cache_clear()
    yield
    devices.xgboost_cuda_available.cache_clear()


@pytest.mark.parametrize(
    "use_cuda, count, expected",
    [(False, 1, False), (True, 0, False), (True, 1, True), (True, 2, True)],
)
def test_xgboost_probe_needs_a_cuda_build_and_a_visible_device(
    monkeypatch, fresh_probe, use_cuda, count, expected
):
    import xgboost

    monkeypatch.setattr(xgboost, "build_info", lambda: {"USE_CUDA": use_cuda})
    monkeypatch.setattr(devices, "cuda_driver_device_count", lambda: count)

    assert devices.xgboost_cuda_available() is expected


def test_xgboost_probe_skips_the_driver_for_a_cpu_only_build(monkeypatch, fresh_probe):
    import xgboost

    monkeypatch.setattr(xgboost, "build_info", lambda: {"USE_CUDA": False})

    def fail():
        raise AssertionError("the driver must not be asked")

    monkeypatch.setattr(devices, "cuda_driver_device_count", fail)
    assert devices.xgboost_cuda_available() is False


def test_driver_count_is_zero_without_a_cuda_driver(monkeypatch):
    def missing(name):
        raise OSError(f"{name}: cannot open shared object file")

    monkeypatch.setattr(devices.ctypes, "CDLL", missing)
    assert devices.cuda_driver_device_count() == 0


def test_resolving_the_xgboost_device_does_not_import_torch(tmp_path):
    """The probe itself stays torch-free (``LibraryModel`` imports torch for its panel)."""
    code = (
        "import sys\n"
        "from quantlab.model.predefined._support.devices import xgboost_default_device\n"
        "assert xgboost_default_device() in ('cuda', 'cpu')\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# --------------------------------------------------------------------------
# XGBTD
# --------------------------------------------------------------------------


def test_xgb_td_passes_no_device(tmp_path, cuda):
    cuda(True)
    head = _head(XGBTDRegressor, tmp_path)
    head._init_model(1, 1, head.config.hyperparameters)

    assert "device" not in head._resolved_hyperparameters()
