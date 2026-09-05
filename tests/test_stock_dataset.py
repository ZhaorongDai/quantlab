"""Unit/integration tests for dataset/stock.py:StockDataset (Phase 2 Plan 07,
DATA-01).

Tests 1-2 cover the dedup_raw_frame() insertion into
StockDataset._raw_data_to_xr() (D-05, Task 1).
Test 3 is a full mocked-network integration test proving the
TiingoAcquisition -> StockDataset -> Zarr round trip end-to-end (Task 2).
"""

from datetime import datetime
from pathlib import Path

import polars as pl

from base.config import AcquisitionConfig, DatasetConfig
from dataset.stock import StockDataset

_COLUMNS = [
    "timestamp",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adjOpen",
    "adjHigh",
    "adjLow",
    "adjClose",
    "adjVolume",
    "divCash",
    "splitFactor",
]


def _row(date_str: str, symbol: str, close: float = 100.0) -> dict:
    # Mirrors the exact field set/dtypes acquisition/tiingo.py:TiingoAcquisition
    # writes (Tiingo's EOD columns + divCash/splitFactor + renamed
    # timestamp/symbol) so synthetic fixtures schema-match real vendor
    # output when pl.concat()'d together in StockDataset._raw_data_to_xr().
    return {
        "timestamp": datetime.fromisoformat(date_str),
        "symbol": symbol,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": 1_000,
        "adjOpen": close,
        "adjHigh": close,
        "adjLow": close,
        "adjClose": close,
        "adjVolume": 1_000,
        "divCash": 0.0,
        "splitFactor": 1.0,
    }


def _write_stock_pqt(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(rows).select(_COLUMNS)
    df.write_parquet(path)
    return path


def _make_dataset_config(raw_data_dir_path: str, zarr_file_path: str) -> DatasetConfig:
    return DatasetConfig(
        raw_data_dir_path=raw_data_dir_path,
        zarr_file_path=zarr_file_path,
        catalog_path=raw_data_dir_path,
        market="us_equity",
        frequency="1d",
    )


def test_overlapping_pqt_files_dedup_before_to_xarray(tmp_path: Path) -> None:
    """Test 1: two raw parquet files for the same symbol, sharing one
    overlapping (timestamp, symbol) row (simulating an overlapping
    download()+refresh() range), convert successfully via
    StockDataset.from_raw_data() without raising ValueError (the
    non-unique-MultiIndex crash), and the resulting dataset has exactly one
    row for the overlapping timestamp."""
    symbol_dir = tmp_path / "raw" / "AAPL"
    _write_stock_pqt(
        symbol_dir / "data_1.pqt",
        [_row("2024-01-02", "AAPL"), _row("2024-01-03", "AAPL")],
    )
    _write_stock_pqt(
        symbol_dir / "data_2.pqt",
        [_row("2024-01-03", "AAPL"), _row("2024-01-04", "AAPL")],
    )

    config = _make_dataset_config(
        str(tmp_path / "raw"), str(tmp_path / "out.zarr")
    )
    dataset = StockDataset(config)
    dataset.from_raw_data()

    xr_data = dataset.get_xarray_dataset()
    # 3 unique timestamps (Jan 2, 3, 4), 1 symbol -- the overlapping Jan 3
    # row collapsed from 2 duplicate rows into exactly 1.
    assert xr_data.sizes["timestamp"] == 3
    assert xr_data.sizes["symbol"] == 1


def test_dedup_noop_on_non_overlapping_pqt_files(
    tmp_path: Path,
) -> None:
    """Test 2: a StockDataset built from non-overlapping parquet files
    converts unchanged (same row count) -- regression proving the dedup
    insertion doesn't alter clean data."""
    symbol_dir = tmp_path / "raw" / "AAPL"
    _write_stock_pqt(
        symbol_dir / "data_1.pqt",
        [_row("2024-01-02", "AAPL"), _row("2024-01-03", "AAPL")],
    )
    _write_stock_pqt(
        symbol_dir / "data_2.pqt",
        [_row("2024-01-04", "AAPL"), _row("2024-01-05", "AAPL")],
    )

    config = _make_dataset_config(
        str(tmp_path / "raw"), str(tmp_path / "out.zarr")
    )
    dataset = StockDataset(config)
    dataset.from_raw_data()

    xr_data = dataset.get_xarray_dataset()
    assert xr_data.sizes["timestamp"] == 4
    assert xr_data.sizes["symbol"] == 1
    assert set(["timestamp", "symbol"]).issubset(set(xr_data.dims))


def test_tiingo_acquisition_to_stock_dataset_zarr_round_trip(
    mock_tiingo_client, tiingo_json_response: list[dict], tmp_path: Path
) -> None:
    """Test 3: TiingoAcquisition.download() (mocked network) ->
    StockDataset.from_raw_data().save() -> StockDataset(...).read() round
    trips a [timestamp, symbol] xr.Dataset through a real (tmp-path) Zarr
    store, proving DATA-01 end-to-end. Only the outermost HTTP call is
    mocked -- every other layer (TiingoAcquisition's raw-parquet write,
    StockDataset's raw-to-xarray conversion + clean_market_data, XrBackend's
    Zarr write/read) runs for real."""
    from acquisition.tiingo import TiingoAcquisition

    raw_data_dir_path = str(tmp_path / "raw")
    zarr_file_path = str(tmp_path / "stock.zarr")

    acq_config = AcquisitionConfig(
        market="us_equity",
        frequency="1d",
        raw_data_dir_path=raw_data_dir_path,
        watermark_path=str(tmp_path / "watermark"),
        symbols=("AAPL",),
        start_date="2024-01-01",
        end_date="2024-01-31",
    )
    TiingoAcquisition(acq_config).download(["AAPL"])

    ds_config = _make_dataset_config(raw_data_dir_path, zarr_file_path)
    StockDataset(ds_config).from_raw_data().save()

    read_back = StockDataset(ds_config).read()
    xr_data = read_back.get_xarray_dataset()

    assert set(["timestamp", "symbol"]).issubset(set(xr_data.dims))
    assert "adjClose" in xr_data.data_vars
    assert "anomaly_flag" in xr_data.data_vars

    expected_close_by_date = {
        row["date"][:10]: row["adjClose"] for row in tiingo_json_response
    }
    for date_str, expected_close in expected_close_by_date.items():
        actual = xr_data["adjClose"].sel(
            timestamp=date_str, symbol="AAPL"
        ).item()
        assert actual == expected_close

    # A (timestamp, symbol) combination absent from the fetched data is NaN
    # in the read-back result -- not forward-filled.
    other_config = _make_dataset_config(
        raw_data_dir_path, str(tmp_path / "stock_multi.zarr")
    )
    other_config.symbols = None
    multi_symbol_dataset = StockDataset(other_config)
    # Write a second symbol's raw data covering a disjoint date so a gap
    # exists for AAPL on that date once both symbols share the same
    # [timestamp, symbol] grid.
    _write_stock_pqt(
        Path(raw_data_dir_path) / "MSFT" / "data_1.pqt",
        [_row("2024-02-01", "MSFT")],
    )
    multi_symbol_dataset.from_raw_data().save()
    read_back_multi = StockDataset(other_config).read()
    xr_multi = read_back_multi.get_xarray_dataset()

    import numpy as np

    aapl_on_gap_date = xr_multi["adjClose"].sel(
        timestamp="2024-02-01", symbol="AAPL"
    ).item()
    assert np.isnan(aapl_on_gap_date)
