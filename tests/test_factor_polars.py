"""Polars factor-backend tests (FACTOR-03).

Scaffolded by 03-01 Task 2 (Nyquist Wave 0). The real content -- the
`FactorPolars` ABC (`base/factor_polars.py`), the `Momentum` example factor
(`factor/momentum.py`, D-08) and the D-04 laziness proof -- lands in **03-04**.

The single test below is an infrastructure self-test: it locks in the raw
column contract `factor/momentum.py` will be written against. `FactorPolars`
consumes `Dataset.get_lazyframe()`, which -- unlike the KunQuant path's
`_to_kunquant()` -- performs **no** rename, so a Polars factor over crypto
spot data sees Binance's raw Title-Case `Close`, not the lowercase `close`
KunQuant receives. Getting that backwards is the most likely way 03-04's
first draft fails.

Import-safety rule (tests/conftest.py module docstring): nothing here may
import `base.factor_polars`, `factor.momentum` or `PolarsFactorConfig` at
module level -- none exists yet.
"""

from typing import Callable

from base.config import DatasetConfig
from dataset.spot import SpotKlineDataset


def test_spot_kline_lazyframe_exposes_raw_title_case_columns(
    spot_kline_zarr: Callable[..., DatasetConfig],
) -> None:
    """`SpotKlineDataset.get_lazyframe()` exposes the `[timestamp, symbol]`
    index columns plus Binance's RAW Title-Case OHLCV names -- the Polars
    backend's boundary contract (FACTOR-03 / D-04), distinct from the
    lowercase names `_to_kunquant()` renames to for KunQuant.
    """
    dataset_config = spot_kline_zarr()
    lazyframe = SpotKlineDataset(dataset_config).read().get_lazyframe()

    names = lazyframe.collect_schema().names()

    assert "timestamp" in names
    assert "symbol" in names
    assert "Close" in names
