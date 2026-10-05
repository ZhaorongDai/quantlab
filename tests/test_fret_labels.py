"""``Return`` and ``BinaryReturn`` are ``Forward`` labels over a trailing open-to-open return.

The label at bar t is ``adjOpen[t + n + 1] / adjOpen[t + 1] - 1`` (``Return``)
or whether it is positive (``BinaryReturn``): a position entered at the next
bar's open and held ``n`` bars. They are labels only, and their config stays
the ``FactorConfig`` they are built from.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from quantlab.base.config import FactorConfig
from quantlab.dataset.config import DatasetConfig
from quantlab.dataset.stock import StockDataset
from quantlab.label.forward import Forward
from quantlab.label.predefined.fret import BinaryReturn, Return
from quantlab.core.component import rebuild

N = 3


def _config(dataset_config: DatasetConfig, tmp_path: Path, name: str) -> FactorConfig:
    return FactorConfig(
        warmup_bars=N + 1,
        dataset=StockDataset(dataset_config),
        mode="batch",
        data_columns=("adjOpen",),
        kwargs={"n_forward_periods": N},
        file_path=str(tmp_path / "label" / f"{name}.zarr"),
        njobs=2,
    )


@pytest.fixture
def dataset_config(stock_zarr):
    """60 daily bars from 2024-01-01 to 2024-02-29."""
    return stock_zarr(periods=60)


def _open_to_open(dataset_config: DatasetConfig) -> np.ndarray:
    """``adjOpen[t + N + 1] / adjOpen[t + 1] - 1`` straight from the store, NaN past its end."""
    opens = (
        xr.open_zarr(dataset_config.zarr_file_path)["adjOpen"]
        .transpose("timestamp", "symbol")
        .values.astype(float)
    )
    out = np.full_like(opens, np.nan)
    out[: -(N + 1)] = opens[N + 1 :] / opens[1 : -N] - 1
    return out


def _values(panel: xr.Dataset, name: str) -> np.ndarray:
    return panel[name].transpose("timestamp", "symbol").values


def test_a_return_is_a_forward_label_n_plus_one_bars_ahead(dataset_config, tmp_path):
    label = Return(_config(dataset_config, tmp_path, "ret"))

    assert isinstance(label, Forward)
    assert label.lookahead_bars() == N + 1
    assert label.span_bars() == N
    assert label.get_factor_names() == (f"ret_{N}",)


def test_a_return_is_the_open_to_open_return_after_the_next_bar(
    dataset_config, tmp_path
):
    label = Return(_config(dataset_config, tmp_path, "ret"))

    got = _values(label.compute("2024-01-01", "2024-02-29"), f"ret_{N}")

    # KunQuant computes in float32.
    np.testing.assert_allclose(got, _open_to_open(dataset_config), rtol=1e-5, atol=1e-6)


def test_a_return_request_ending_inside_the_data_has_no_nan_tail(
    dataset_config, tmp_path
):
    label = Return(_config(dataset_config, tmp_path, "ret"))

    got = _values(label.compute("2024-01-10", "2024-01-31"), f"ret_{N}")

    assert not np.isnan(got).any()


def test_a_binary_return_marks_positive_open_to_open_returns(dataset_config, tmp_path):
    label = BinaryReturn(_config(dataset_config, tmp_path, "up"))
    expected = _open_to_open(dataset_config)

    got = _values(label.compute("2024-01-01", "2024-02-29"), f"ret_binary_{N}")

    known = ~np.isnan(expected)
    np.testing.assert_array_equal(got[known], (expected[known] > 0).astype(float))
    assert np.isnan(got[~known]).all()


def test_a_return_reads_its_own_store(dataset_config, tmp_path):
    label = Return(_config(dataset_config, tmp_path, "ret"))
    label.build("2024-01-01", "2024-02-29")

    xr.testing.assert_allclose(
        label.read("2024-01-10", "2024-01-31"),
        label.compute("2024-01-10", "2024-01-31"),
    )


@pytest.mark.parametrize("cls", [Return, BinaryReturn])
def test_a_return_label_rebuilds_from_its_factor_config(cls, dataset_config, tmp_path):
    label = cls(_config(dataset_config, tmp_path, "ret"))

    config = json.loads(json.dumps(label.get_config()))
    rebuilt = rebuild(config)

    assert config["name"] == f"quantlab.label.predefined.fret.{cls.__name__}"
    assert config["kwargs"] == {"n_forward_periods": N}
    assert "dataset" in config and "factor" not in config
    assert rebuilt == label


@pytest.fixture
def ragged_dataset_config(stock_zarr):
    """60 daily bars with interior NaN opens, a late listing and an early delisting.

    ``AAPL`` misses two interior opens, ``MSFT`` lists on bar 15 and ``NVDA``
    delists after bar 40.
    """
    config = stock_zarr(symbols=["AAPL", "MSFT", "NVDA"], periods=60)
    panel = xr.open_zarr(config.zarr_file_path).load()
    opens = panel["adjOpen"].transpose("timestamp", "symbol").values.copy()
    opens[[20, 33], 0] = np.nan
    opens[:15, 1] = np.nan
    opens[41:, 2] = np.nan
    panel["adjOpen"] = (("timestamp", "symbol"), opens)
    panel.to_zarr(config.zarr_file_path, mode="w")
    return config


def test_a_binary_return_is_nan_wherever_the_return_is_nan(
    ragged_dataset_config, tmp_path
):
    ret = Return(_config(ragged_dataset_config, tmp_path, "ret"))
    up = BinaryReturn(_config(ragged_dataset_config, tmp_path, "up"))

    returns = _values(ret.compute("2024-01-01", "2024-02-29"), f"ret_{N}")
    got = _values(up.compute("2024-01-01", "2024-02-29"), f"ret_binary_{N}")

    missing = np.isnan(returns)
    # Interior gaps, the late listing and the early delisting all leave holes
    # before the forward-shift tail.
    assert missing[: -(N + 1)].any(axis=0).all()
    np.testing.assert_array_equal(np.isnan(got), missing)
    np.testing.assert_array_equal(
        got[~missing], (returns[~missing] > 0).astype(float)
    )
