"""`from_raw_data()` converts exactly once per call, with no hidden caching
(quick task 260907-1du).

`from_raw_data()` has no construction-time fallback and no dedup handoff --
constructing a `BaseDataset` subclass never touches the raw source or the
Zarr store, and every explicit call to `from_raw_data()` reconverts
unconditionally. These tests pin exactly that: the ingest idiom
`Dataset(cfg).from_raw_data().save()` (the old Alpaca/Tiingo download scripts)
converts the raw tree once, and calling `from_raw_data()` a second time
converts it again rather than silently reusing the first result.
"""

from pathlib import Path
from typing import Callable

import pytest
import xarray as xr

from quantlab.base.config import DatasetConfig
from quantlab.dataset.stock import StockDataset


def _make_dataset_config(
    raw_data_dir_path: str,
    zarr_file_path: str,
    symbols: tuple[str, ...],
    vendor: str = "tiingo",
) -> DatasetConfig:
    """Same shape as tests/test_stock_dataset.py:_make_dataset_config, with
    `symbols` PINNED to match the shape the ingest scripts construct.
    """
    return DatasetConfig(
        raw_data_dir_path=raw_data_dir_path,
        zarr_file_path=zarr_file_path,
        market="us_equity",
        frequency="1d",
        vendor=vendor,  # type: ignore[arg-type]
        symbols=symbols,
    )


@pytest.fixture
def count_raw_conversions(monkeypatch) -> Callable[[], int]:
    """Install a call counter on `StockDataset._raw_data_to_xr`.

    Counts CONVERSIONS, not wall-clock time: a timing assertion would pass on
    a fast machine for the wrong reason.
    """
    calls: list[str] = []
    original = StockDataset._raw_data_to_xr

    def _counted(self) -> xr.Dataset:
        calls.append("call")
        return original(self)

    monkeypatch.setattr(StockDataset, "_raw_data_to_xr", _counted)
    return lambda: len(calls)


def _ingest_config(
    stock_pqt_row: Callable[..., dict],
    hive_raw_tree: Callable[..., Path],
    tmp_path: Path,
) -> DatasetConfig:
    """A pinned-symbol config over a raw tree with NO destination store -- the
    exact shape the old Alpaca/Tiingo download scripts constructed.
    """
    hive_raw_tree(
        tmp_path / "raw",
        "tiingo",
        [
            stock_pqt_row("2024-01-02", "AAPL"),
            stock_pqt_row("2024-01-03", "AAPL"),
            stock_pqt_row("2024-02-01", "AAPL"),
        ],
    )
    return _make_dataset_config(
        str(tmp_path / "raw" / "tiingo"),
        str(tmp_path / "out.zarr"),
        symbols=("AAPL",),
    )


def test_one_ingest_runs_exactly_one_raw_conversion(
    stock_pqt_row: Callable[..., dict],
    hive_raw_tree: Callable[..., Path],
    tmp_path: Path,
    count_raw_conversions: Callable[[], int],
) -> None:
    """`StockDataset(cfg).from_raw_data().save()` converts raw exactly once:
    construction itself performs no conversion, so the explicit
    `from_raw_data()` call is the only one that runs.
    """
    config = _ingest_config(stock_pqt_row, hive_raw_tree, tmp_path)

    dataset = StockDataset(config).from_raw_data()
    dataset.save()

    assert count_raw_conversions() == 1
    assert Path(config.zarr_file_path).exists()
    assert xr.open_zarr(config.zarr_file_path).sizes["timestamp"] == 3


def test_a_second_from_raw_data_call_reconverts(
    stock_pqt_row: Callable[..., dict],
    hive_raw_tree: Callable[..., Path],
    tmp_path: Path,
    count_raw_conversions: Callable[[], int],
) -> None:
    """`from_raw_data()` caches nothing: calling it twice converts twice,
    even with no change to the config or the backend's held panel in
    between.
    """
    config = _ingest_config(stock_pqt_row, hive_raw_tree, tmp_path)

    dataset = StockDataset(config).from_raw_data()
    assert count_raw_conversions() == 1

    dataset.from_raw_data()

    assert count_raw_conversions() == 2
