"""Unit tests for dataset/spot.py:SpotKlineDataset and ingest_binance_spot.py
(Phase 2 Plan 05, CONTEXT.md D-04/D-05).

Tests 1-2 cover the dedup_raw_frame() insertion into
SpotKlineDataset._raw_data_to_xr() (D-05, Task 1).
Tests 3-4 cover ingest_binance_spot.py:_build_dataset_config()'s
--raw-data-dir override behavior (Task 2).
"""

from typing import Callable

from base.config import DatasetConfig
from dataset.spot import SpotKlineDataset


def _make_config(raw_data_dir_path: str) -> DatasetConfig:
    return DatasetConfig(
        raw_data_dir_path=raw_data_dir_path,
        zarr_file_path=str(
            __import__("pathlib").Path(raw_data_dir_path).parent / "klines.zarr"
        ),
        catalog_path=raw_data_dir_path,
        market="crypto_spot",
        frequency="1d",
    )


def test_overlapping_monthly_csvs_dedup_before_to_xarray(
    write_binance_csv: Callable[..., "__import__('pathlib').Path"],
    binance_csv_rows: list[list],
    tmp_path,
) -> None:
    """Test 1: two monthly CSV files whose date ranges overlap by one day
    (the same (timestamp, symbol) row present in both files, simulating the
    real-world Binance monthly-CSV-near-month-boundary duplicate risk) no
    longer raise `ValueError: cannot convert a DataFrame with a non-unique
    MultiIndex into xarray` -- SpotKlineDataset.from_raw_data() succeeds and
    the resulting dataset has exactly one row per overlapping
    (timestamp, symbol) pair."""
    day1, day2 = binance_csv_rows[0], binance_csv_rows[1]
    day3 = list(day2)
    day3[0] = day2[0] + 86400000  # one day after day2, still ms epoch

    # File 1 (January): day1, day2. File 2 (February): day2 (overlap), day3.
    write_binance_csv(symbol="BTCUSDT", year_month="2024-01", rows=[day1, day2])
    write_binance_csv(symbol="BTCUSDT", year_month="2024-02", rows=[day2, day3])

    config = _make_config(str(tmp_path))
    dataset = SpotKlineDataset(config)

    dataset.from_raw_data()

    xr_data = dataset.get_xarray_dataset()
    # 3 unique timestamps (day1, day2, day3), 1 symbol -- the overlapping
    # day2 row collapsed from 2 duplicate rows into exactly 1.
    assert xr_data.sizes["timestamp"] == 3
    assert xr_data.sizes["symbol"] == 1


def test_non_duplicate_csvs_convert_with_unchanged_row_count(
    write_binance_csv: Callable[..., "__import__('pathlib').Path"],
    tmp_path,
) -> None:
    """Test 2: a SpotKlineDataset built from CSVs with no duplicate rows
    converts to an xr.Dataset with the expected [timestamp, symbol] dims and
    an unchanged row count (regression -- dedup insertion must not alter
    non-duplicate data)."""
    write_binance_csv(symbol="BTCUSDT", year_month="2024-01")

    config = _make_config(str(tmp_path))
    dataset = SpotKlineDataset(config)

    dataset.from_raw_data()

    xr_data = dataset.get_xarray_dataset()
    assert xr_data.sizes["timestamp"] == 2
    assert xr_data.sizes["symbol"] == 1
    assert set(["timestamp", "symbol"]).issubset(set(xr_data.dims))
