"""`FrameDataset`: a caller's frame or panel held in memory, used as a `MarketDataset`.

The secondary seam of ADR 0011: behaviour `quantlab.api` cannot show. A factor built
directly on a `FrameDataset` computes the same panel as on the equivalent Zarr-backed
dataset, the dataset answers the read requests every consumer makes (`panel`,
`bar_before`, `head`, `to_kunquant`) from memory, and it refuses what it cannot do.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr

from conftest import compute_all
from quantlab.base.config import FactorConfig
from quantlab.dataset.memory import FrameDataset
from quantlab.dataset.stock import StockDataset
from quantlab.factor.predefined.alpha158 import Alpha158Stock
from tests.backtest_fixtures import ADJUSTED_COLUMNS, write_price_store


@pytest.fixture
def stock(tmp_path) -> StockDataset:
    return StockDataset(write_price_store(tmp_path))


@pytest.fixture
def store(stock) -> xr.Dataset:
    return xr.open_zarr(stock.config.zarr_file_path).load()


def _alpha158(dataset) -> xr.Dataset:
    return compute_all(
        Alpha158Stock(
            FactorConfig(
                warmup_bars=0,
                dataset=dataset,
                mode="batch",
                data_columns=ADJUSTED_COLUMNS,
                njobs=4,
            )
        )
    )


def test_alpha158_on_a_frame_dataset_equals_the_zarr_backed_one(stock, store):
    frame = store[list(ADJUSTED_COLUMNS)].to_dataframe().reset_index()

    on_frame = _alpha158(FrameDataset(frame))

    xr.testing.assert_equal(on_frame, _alpha158(stock))


def test_a_frame_dataset_built_from_a_panel_computes_the_same(stock, store):
    xr.testing.assert_equal(_alpha158(FrameDataset(store)), _alpha158(stock))


def test_columns_renames_the_frame_before_it_is_held(store):
    frame = store[["adjClose"]].to_dataframe().reset_index()
    frame = frame.rename(columns={"timestamp": "date", "symbol": "ticker", "adjClose": "px"})

    dataset = FrameDataset(
        pl.from_pandas(frame), columns={"date": "timestamp", "ticker": "symbol", "px": "close"}
    )

    panel = dataset.panel("2024-01-01", "2024-12-31")
    np.testing.assert_array_equal(panel["close"].values, store["adjClose"].values)


def test_read_requests_are_answered_from_memory(stock, store):
    dataset = FrameDataset(store)

    xr.testing.assert_equal(
        dataset.panel("2024-01-03", "2024-01-10", symbols=["CCC", "AAA"]),
        stock.panel("2024-01-03", "2024-01-10", symbols=["CCC", "AAA"]).load(),
    )
    assert dataset.bar_before("2024-01-08", 1) == stock.bar_before("2024-01-08", 1)
    assert dataset.bar_after("2024-01-05", 1) == stock.bar_after("2024-01-05", 1)
    assert dataset.head(2).collect().shape[0] == 2
    assert set(dataset.head(2).collect().columns) >= {"timestamp", "symbol", "adjClose"}

    inputs, symbols, timestamps = dataset.to_kunquant(
        ("adjClose",), panel=dataset.panel("2024-01-01", "2024-12-31")
    )
    expected, _, _ = stock.to_kunquant(
        ("adjClose",), panel=stock.panel("2024-01-01", "2024-12-31")
    )
    np.testing.assert_array_equal(inputs["adjClose"], expected["adjClose"])
    assert inputs["adjClose"].dtype == np.float32


def test_too_little_history_raises_the_library_error(store):
    from quantlab.base.data import InsufficientHistoryError

    with pytest.raises(InsufficientHistoryError, match="in memory"):
        FrameDataset(store).bar_before("2024-01-03", 5)


def test_resample_aggregates_in_memory_and_writes_nothing(tmp_path):
    timestamps = pd.date_range("2024-01-01", periods=48, freq="h")
    frame = pd.DataFrame(
        {
            "timestamp": np.repeat(timestamps, 2),
            "symbol": ["AAA", "BBB"] * 48,
            "close": np.arange(96, dtype=float),
        }
    )

    daily = FrameDataset(frame).resample("1d", "last")

    panel = daily.panel("2024-01-01", "2024-01-02")
    assert panel.sizes["timestamp"] == 2
    np.testing.assert_array_equal(panel["close"].sel(symbol="AAA").values, [46.0, 94.0])
    assert daily.bar_before("2024-01-02", 1) == pd.Timestamp("2024-01-01")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "call",
    [
        lambda d: d.from_raw_data(),
        lambda d: d.from_raw_data_chunked(),
        lambda d: d.update(),
        lambda d: d.save(),
    ],
    ids=["from_raw_data", "from_raw_data_chunked", "update", "save"],
)
def test_building_or_saving_is_refused(store, call):
    with pytest.raises(ValueError, match="FrameDataset.*in memory"):
        call(FrameDataset(store))


def test_a_stream_mode_factor_refuses_a_frame_dataset(store):
    with pytest.raises(ValueError, match="stream.*FrameDataset"):
        Alpha158Stock(
            FactorConfig(
                warmup_bars=0,
                dataset=FrameDataset(store),
                mode="stream",
                data_columns=ADJUSTED_COLUMNS,
                factor_names=["KMID"],
            )
        )


def test_datasets_holding_different_data_are_not_equal(store):
    assert FrameDataset(store) == FrameDataset(store.copy())
    assert FrameDataset(store) != FrameDataset(store * 2)
