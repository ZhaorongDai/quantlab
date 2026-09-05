"""End-to-end batch KunQuant factor-computation tests (Phase 3 Plan 01,
FACTOR-01 / ROADMAP Phase 3 Success Criterion 1).

Before 03-01, BOTH tests in this file raised
`TypeError: WindowedZScore.decompose() takes 1 positional argument but 2 were
given`: `my_ops/preprocess.py` declared `decompose(self)` while the installed
KunQuant 0.1.11 declares the contract as
`CompositiveOp.decompose(self, options: dict)` (`KunQuant/Op.py:292`) and
invokes it positionally (`KunQuant/passes/Decompose.py:15`). Because both
`Alpha101SpotKline` and `Alpha158SpotKline` wrap every `Output(...)` in
`WindowedZScore(...)`, every batch `.cal()` in the repository was dead. These
tests are the regression lock on that fix.

Cost control: each test passes an explicit 1-3 element `factor_names` list so
the compiled graph stays tiny (~1.5 s per compilation) and `njobs=4` so the
KunQuant executor does not spawn `FactorConfig`'s default 128 threads.
"""

from pathlib import Path
from typing import Callable

import numpy as np
import xarray as xr

from base.config import DatasetConfig, FactorConfig
from dataset.spot import SpotKlineDataset
from factor.alpha101 import Alpha101SpotKline
from factor.alpha158 import Alpha158SpotKline


def _factor_config(
    dataset_config: DatasetConfig,
    factor_names: list[str],
    data_columns: list[str],
    tmp_path: Path,
    window: int = 10,
) -> FactorConfig:
    """Build a `FactorConfig` over a `SpotKlineDataset` for the given synthetic
    Zarr store.

    `factor_names` is always passed explicitly: it both bypasses the
    `FactorKunQuant.config` setter's `_get_factor_names()` call (which would
    enumerate all 169 Alpha158 / 82 Alpha101 names) and restricts the compiled
    graph to only the reachable `Output(...)` nodes.
    """
    return FactorConfig(
        window=window,
        dataset=SpotKlineDataset(dataset_config),
        mode="batch",
        data_columns=data_columns,
        factor_names=factor_names,
        file_path=str(tmp_path / "factors" / "out.zarr"),
        njobs=4,
    )


def test_alpha158_spot_batch_cal_returns_xarray_dataset(
    spot_kline_zarr: Callable[..., DatasetConfig], tmp_path: Path
) -> None:
    """FACTOR-01 / ROADMAP Phase 3 Success Criterion 1: computing the Alpha158
    factor set in batch mode over crypto spot data returns an `xr.Dataset`
    indexed by `[timestamp, symbol]` carrying real, finite factor values.
    """
    dataset_config = spot_kline_zarr(periods=60, seed=0)
    factor = Alpha158SpotKline(
        _factor_config(
            dataset_config,
            factor_names=["KMID", "VOLUME0", "STD5"],
            data_columns=["open", "close", "volume"],
            tmp_path=tmp_path,
        )
    )

    result = factor.cal().get_features()

    assert isinstance(result, xr.Dataset)
    assert dict(result.sizes) == {"timestamp": 60, "symbol": 8}
    assert sorted(result.data_vars) == ["KMID", "STD5", "VOLUME0"]
    assert np.isfinite(result["KMID"].to_numpy()).sum() > 0


def test_alpha101_spot_batch_cal_returns_xarray_dataset(
    spot_kline_zarr: Callable[..., DatasetConfig], tmp_path: Path
) -> None:
    """FACTOR-01: the same batch path works for the Alpha101 family, proving
    the shared `WindowedZScore` normalization op decomposes correctly for both
    alpha families rather than only the one that happened to be exercised.
    """
    dataset_config = spot_kline_zarr(periods=60, seed=0)
    factor = Alpha101SpotKline(
        _factor_config(
            dataset_config,
            factor_names=["alpha001"],
            data_columns=["close"],
            tmp_path=tmp_path,
        )
    )

    result = factor.cal().get_features()

    assert isinstance(result, xr.Dataset)
    assert "alpha001" in result.data_vars
    assert np.isfinite(result["alpha001"].to_numpy()).sum() > 0
