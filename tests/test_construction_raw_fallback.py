"""Construction-time store probe and the raw fallback it guards (quick task
260907-1du).

Every dataset config in this module sets `symbols=(...)` NON-None. That is
what makes `BaseDataset._reset_symbols()` fire at all, and every pre-existing
fixture in this repo leaves `symbols` unset -- which is exactly why a green
suite could not see either defect these tests pin:

- a present-but-EMPTY destination store crashed construction with
  `ValueError: could not convert string to float`, because the fallback keys
  off `FileNotFoundError` from `read()` and an empty store does not raise
  there;
- one ingest ran the raw-to-xarray conversion TWICE, because the
  construction-time fallback materialises the whole panel and the caller then
  asks for the same panel again microseconds later.

loguru does not propagate to stdlib `logging`, so warnings are captured with a
temporary in-memory sink rather than `caplog` (same idiom as
tests/test_chunked_ingest.py).
"""

from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import xarray as xr
from loguru import logger

from base.config import DatasetConfig
from dataset.stock import StockDataset

_OHLCV = ("open", "high", "low", "close", "volume")


def _make_dataset_config(
    raw_data_dir_path: str,
    zarr_file_path: str,
    symbols: tuple[str, ...],
    vendor: str = "tiingo",
) -> DatasetConfig:
    """Same shape as tests/test_stock_dataset.py:_make_dataset_config, with
    `symbols` PINNED -- the one difference that makes `_reset_symbols()` run.
    """
    return DatasetConfig(
        raw_data_dir_path=raw_data_dir_path,
        zarr_file_path=zarr_file_path,
        catalog_path=raw_data_dir_path,
        market="us_equity",
        frequency="1d",
        vendor=vendor,  # type: ignore[arg-type]
        symbols=symbols,
    )


def _write_empty_store(path: Path) -> Path:
    """Write the exact store shape a zero-row panel leaves behind: present on
    disk, five zero-shaped OHLCV variables, and both coordinates length 0.
    """
    empty = xr.Dataset(
        {
            name: (["timestamp", "symbol"], np.zeros((0, 0), dtype="float64"))
            for name in _OHLCV
        },
        coords={
            "timestamp": np.array([], dtype="datetime64[ns]"),
            "symbol": np.array([], dtype=object),
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    empty.to_zarr(path, mode="w")
    return path


def _write_populated_store(path: Path, close: float) -> Path:
    values = np.full((2, 1), close, dtype="float64")
    populated = xr.Dataset(
        {name: (["timestamp", "symbol"], values.copy()) for name in _OHLCV},
        coords={
            "timestamp": np.array(
                ["2024-01-02", "2024-01-03"], dtype="datetime64[ns]"
            ),
            "symbol": ["AAPL"],
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    populated.to_zarr(path, mode="w")
    return path


def _captured_warnings():
    messages: list[str] = []
    sink_id = logger.add(messages.append, level="WARNING", format="{message}")
    return messages, sink_id


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


# ---------------------------------------------------------------------------
# Defect 1 -- reach the raw fallback when the store is present but EMPTY
# ---------------------------------------------------------------------------


def test_a_present_but_empty_store_falls_back_to_raw(
    stock_pqt_row: Callable[..., dict],
    hive_raw_tree: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """A leftover zero-row store must route to the SAME `from_raw_data()`
    fallback a missing store already takes, instead of crashing construction.

    Reddening mutation: remove the pre-probe from `_reset_symbols()` -- the
    `.sel({"symbol": [...]})` inside `read()` then raises
    `ValueError: could not convert string to float`.
    """
    store_path = _write_empty_store(tmp_path / "out.zarr")

    # Assert the FIXTURE reproduces the measured store rather than merely
    # resembling it: zarr reads a zero-length object coordinate back as
    # float64, and it is that float64 axis `.sel` fails to convert 'AAPL' to.
    reopened = xr.open_zarr(store_path)
    try:
        assert reopened["symbol"].dtype == np.dtype("float64")
        assert reopened["symbol"].size == 0
    finally:
        reopened.close()

    hive_raw_tree(
        tmp_path / "raw",
        "tiingo",
        [stock_pqt_row("2024-01-02", "AAPL"), stock_pqt_row("2024-01-03", "AAPL")],
    )
    config = _make_dataset_config(
        str(tmp_path / "raw" / "tiingo"), str(store_path), symbols=("AAPL",)
    )

    dataset = StockDataset(config)

    assert dataset.config.symbols == ("AAPL",)
    assert dataset.get_xarray_dataset().sizes["timestamp"] == 2


def test_an_absent_store_still_falls_back_to_raw(
    stock_pqt_row: Callable[..., dict],
    hive_raw_tree: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """The pre-existing missing-store fallback is unchanged, warning included.

    Reddening mutation: delete the fallback entirely -- construction then
    raises `FileNotFoundError` instead of reading raw.
    """
    hive_raw_tree(
        tmp_path / "raw",
        "tiingo",
        [stock_pqt_row("2024-01-02", "AAPL"), stock_pqt_row("2024-01-03", "AAPL")],
    )
    config = _make_dataset_config(
        str(tmp_path / "raw" / "tiingo"),
        str(tmp_path / "absent.zarr"),
        symbols=("AAPL",),
    )

    messages, sink_id = _captured_warnings()
    try:
        dataset = StockDataset(config)
    finally:
        logger.remove(sink_id)

    assert dataset.config.symbols == ("AAPL",)
    assert dataset.get_xarray_dataset().sizes["timestamp"] == 2
    assert any(
        "data not found, try to read from csv" in message
        for message in messages
    ), messages


def test_a_populated_store_is_still_read_and_no_raw_conversion_runs(
    stock_pqt_row: Callable[..., dict],
    hive_raw_tree: Callable[..., Path],
    tmp_path: Path,
    count_raw_conversions: Callable[[], int],
) -> None:
    """A populated store is still READ, and the raw tree is never touched.

    The store's `close` is 100.0 and the raw tree's is 1.0, so the two sources
    are distinguishable by value as well as by call count.

    Reddening mutation: force the probe branch to always fall back -- the
    conversion count becomes 1 and `close` becomes 1.0.
    """
    store_path = _write_populated_store(tmp_path / "out.zarr", close=100.0)
    hive_raw_tree(
        tmp_path / "raw",
        "tiingo",
        [stock_pqt_row("2024-01-02", "AAPL", close=1.0)],
    )
    config = _make_dataset_config(
        str(tmp_path / "raw" / "tiingo"), str(store_path), symbols=("AAPL",)
    )

    dataset = StockDataset(config)

    assert count_raw_conversions() == 0
    assert dataset.config.symbols == ("AAPL",)
    assert float(dataset.get_xarray_dataset()["close"].values[0, 0]) == 100.0
