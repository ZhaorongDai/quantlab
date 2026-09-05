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
from base.data import Dataset
from dataset.spot import SpotKlineDataset
from dataset.stock import StockDataset
from factor.alpha101 import Alpha101SpotKline, Alpha101Stock
from factor.alpha158 import Alpha158SpotKline


def _factor_config(
    dataset_config: DatasetConfig,
    factor_names: list[str],
    data_columns: list[str],
    tmp_path: Path,
    window: int = 10,
    dataset_cls: type[Dataset] = SpotKlineDataset,
) -> FactorConfig:
    """Build a `FactorConfig` over the given synthetic Zarr store.

    `factor_names` is always passed explicitly: it both bypasses the
    `FactorKunQuant.config` setter's `_get_factor_names()` call (which would
    enumerate all 169 Alpha158 / 82 Alpha101 names) and restricts the compiled
    graph to only the reachable `Output(...)` nodes.

    `dataset_cls` selects the `Dataset` subclass wrapped around
    `dataset_config`. It defaults to `SpotKlineDataset` so every 03-01 caller
    is unaffected; 03-03's US-equity tests pass `StockDataset` rather than
    duplicating this helper.
    """
    return FactorConfig(
        window=window,
        dataset=dataset_cls(dataset_config),
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


# ---------------------------------------------------------------------------
# Phase 3 Plan 03 -- US equities (D-01, D-02, NORM-01/D-09)
#
# Tiingo supplies no dollar-volume column, so `StockDataset._to_kunquant()`
# synthesizes `amount = volume * close` (D-02). Without it every KunQuant
# factor class reading US-equity data dies at graph-construction time:
# `Alpha101.AllData.__init__` unconditionally builds `Div(self.amount, ...)`
# for `vwap`, raising `RuntimeError: Bad inputs, given <class 'NoneType'>`.
#
# `stock_zarr` defaults to 2 symbols, but KunQuant's compiled TS layout
# requires the symbol axis to be a multiple of its SIMD block width (8) or
# `kr.runGraph` raises `RuntimeError: Bad shape at open`. Every test that
# actually COMPILES a graph therefore passes `_STOCK_SYMBOLS` (8 tickers);
# the two `to_kunquant()` tests below never reach KunQuant and use the
# fixture default.
# ---------------------------------------------------------------------------

_STOCK_SYMBOLS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "AVGO",
]


def test_stock_to_kunquant_synthesizes_amount_as_adjusted_dollar_volume(
    stock_zarr: Callable[..., DatasetConfig],
) -> None:
    """D-02: asking `StockDataset.to_kunquant()` for `amount` on a store that
    has no such column yields the `volume * close` dollar-volume proxy,
    computed from the ADJUSTED series (the `adj*` rename runs first), while
    every other requested array is untouched.
    """
    dataset_config = stock_zarr(periods=60, seed=0)

    input_dict, symbols, timestamp = StockDataset(dataset_config).to_kunquant(
        ("open", "close", "volume", "amount")
    )

    assert sorted(input_dict) == ["amount", "close", "open", "volume"]
    np.testing.assert_allclose(
        input_dict["amount"],
        input_dict["volume"] * input_dict["close"],
        rtol=1e-5,
    )
    assert input_dict["amount"].shape == (len(timestamp), len(symbols))
    assert np.isfinite(input_dict["amount"]).all()


def test_stock_to_kunquant_without_amount_leaves_arrays_unchanged(
    stock_zarr: Callable[..., DatasetConfig],
) -> None:
    """D-02 / T-03-03-01: the synthesis is inert when `amount` is not
    requested -- the returned dict carries no `amount` key and every other
    array is byte-identical to the one produced when `amount` IS requested,
    so the guard cannot perturb the pre-existing path.
    """
    dataset_config = stock_zarr(periods=60, seed=0)

    without = StockDataset(dataset_config).to_kunquant(
        ("open", "close", "volume")
    )[0]
    with_amount = StockDataset(dataset_config).to_kunquant(
        ("open", "close", "volume", "amount")
    )[0]

    assert "amount" not in without
    assert sorted(without) == ["close", "open", "volume"]
    for col in ("open", "close", "volume"):
        np.testing.assert_array_equal(without[col], with_amount[col])


def test_alpha101_stock_bugfix_batch_cal_returns_xarray_dataset(
    stock_zarr: Callable[..., DatasetConfig], tmp_path: Path
) -> None:
    """D-02 regression lock: before 03-03 this exact construction raised
    `RuntimeError: Bad inputs, given <class 'NoneType'>` inside
    `Alpha101.AllData.__init__` -- `Alpha101Stock._get_factor_func()` never
    passed `amount`, yet `AllData` unconditionally builds `vwap` from it. With
    the `Input("amount")` node wired in and `StockDataset._to_kunquant()`
    feeding it, the US-equity Alpha101 batch path computes real values.

    Note the absence of any normalization assertion: `Alpha101Stock` emits raw
    factor values by design (NORM-01 / D-09, locked below).
    """
    dataset_config = stock_zarr(symbols=_STOCK_SYMBOLS, periods=60, seed=0)
    factor = Alpha101Stock(
        _factor_config(
            dataset_config,
            factor_names=["alpha001"],
            data_columns=["open", "high", "low", "close", "volume", "amount"],
            tmp_path=tmp_path,
            dataset_cls=StockDataset,
        )
    )

    result = factor.cal().get_features()

    assert isinstance(result, xr.Dataset)
    assert "alpha001" in result.data_vars
    assert np.isfinite(result["alpha001"].to_numpy()).sum() > 0
