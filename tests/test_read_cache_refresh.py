"""Re-dated reads need an explicit refresh: the read-cache trap (D-14, RESEARCH Pitfall 1).

`XrBackend.read(path, overwrite=False)` returns early as soon as the backend
already holds data. Both read paths then narrow that cached panel IN PLACE:
`BaseDataset.read()` calls `_filter()`, and `Factor.read()` calls
`_auto_filter()`. So once an object has been read over a narrow window,
widening `config.start_date` and calling `read()` again returns the same narrow
panel. Nothing raises, and the missing bars simply are not there.

The backtester hits this directly. D-14 re-dates the factors and datasets a
model was trained on, widening their window to cover the backtest plus the
factor warm-up, and then reads again. Without a refresh the warm-up silently
shrinks or the prediction window comes back empty. The RESEARCH probe measured
it: a 10-bar store read from a later start gave 6 bars, still 6 after widening,
and 10 only with `overwrite=True`.

What is locked here, in both directions:

- the default `read()` keeps the cached narrow window, so changing that default
  is a visible decision rather than a side effect;
- `Factor.read(overwrite=True)` re-opens the store and honours widened dates;
- `BaseDataset.read(overwrite=True)` does the same, through the `**kwargs` it
  already forwards.

Dates are written onto the config as ISO `YYYY-MM-DD` strings directly: the
dataset config setter normalizes dates only when a whole config is assigned.
Configs are constructed directly, never through `quantlab/config` factories
(D-32).
"""

import dataclasses
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import xarray as xr

from quantlab.base.config import DatasetConfig, PolarsFactorConfig
from quantlab.dataset.spot import SpotKlineDataset
from quantlab.factor.momentum import Momentum

NARROW_START = "2024-02-10"
WIDE_START = "2024-01-20"
END = "2024-02-29"


def _momentum(
    dataset_config: DatasetConfig,
    tmp_path: Path,
    start_date: str | None = None,
    end_date: str | None = None,
) -> Momentum:
    """Each factor gets its own copy of the dataset config, because the factor
    config setter writes its widened window onto the dataset config."""
    return Momentum(
        PolarsFactorConfig(
            window=5,
            dataset=SpotKlineDataset(dataclasses.replace(dataset_config)),
            file_path=str(tmp_path / "factors" / "momentum.zarr"),
            start_date=start_date,
            end_date=end_date,
            kwargs={"n": 5},
        )
    )


@pytest.fixture
def momentum_store(
    spot_kline_zarr: Callable[..., DatasetConfig], tmp_path: Path
) -> DatasetConfig:
    """Write a Momentum factor store over the whole 60-bar synthetic range."""
    dataset_config = spot_kline_zarr()
    _momentum(dataset_config, tmp_path).cal().save(mode="w")
    return dataset_config


def _timestamps(obj) -> np.ndarray:
    return obj.data_backend.get_xarray_dataset().timestamp.values


def test_factor_read_keeps_the_cached_narrow_window_by_default(
    momentum_store: DatasetConfig, tmp_path: Path
) -> None:
    factor = _momentum(momentum_store, tmp_path, NARROW_START, END).read()
    narrow = len(_timestamps(factor))
    assert 0 < narrow < 60

    factor.config.start_date = WIDE_START

    assert len(_timestamps(factor.read())) == narrow


def test_factor_read_overwrite_true_honours_widened_dates(
    momentum_store: DatasetConfig, tmp_path: Path
) -> None:
    factor = _momentum(momentum_store, tmp_path, NARROW_START, END).read()
    narrow = len(_timestamps(factor))

    factor.config.start_date = WIDE_START
    widened = factor.read(overwrite=True)

    fresh = _momentum(momentum_store, tmp_path, WIDE_START, END).read()
    assert len(_timestamps(widened)) > narrow
    xr.testing.assert_identical(
        widened.data_backend.get_xarray_dataset(),
        fresh.data_backend.get_xarray_dataset(),
    )


def test_dataset_read_overwrite_true_honours_widened_dates(
    spot_kline_zarr: Callable[..., DatasetConfig],
) -> None:
    dataset_config = spot_kline_zarr()
    dataset = SpotKlineDataset(
        dataclasses.replace(dataset_config, start_date=NARROW_START, end_date=END)
    ).read()
    narrow = len(_timestamps(dataset))
    assert 0 < narrow < 60

    dataset.config.start_date = WIDE_START
    assert len(_timestamps(dataset.read())) == narrow

    fresh = SpotKlineDataset(
        dataclasses.replace(dataset_config, start_date=WIDE_START, end_date=END)
    ).read()
    widened = len(_timestamps(dataset.read(overwrite=True)))
    assert widened > narrow
    assert widened == len(_timestamps(fresh))
