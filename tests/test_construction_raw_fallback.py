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

from quantlab.base.config import DatasetConfig
from quantlab.dataset.stock import StockDataset

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






# ---------------------------------------------------------------------------
# Defect 2 -- one ingest, exactly one raw-to-xarray conversion
# ---------------------------------------------------------------------------


def _ingest_config(
    stock_pqt_row: Callable[..., dict],
    hive_raw_tree: Callable[..., Path],
    tmp_path: Path,
) -> DatasetConfig:
    """A pinned-symbol config over a raw tree with NO destination store -- the
    exact shape `ingest_alpaca.py` / `ingest_tiingo.py` construct.
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


def _substitute_panel() -> xr.Dataset:
    values = np.full((1, 1), 7.0, dtype="float64")
    return xr.Dataset(
        {name: (["timestamp", "symbol"], values.copy()) for name in _OHLCV},
        coords={
            "timestamp": np.array(["2024-01-02"], dtype="datetime64[ns]"),
            "symbol": ["ZZZZ"],
        },
    )


def test_one_ingest_runs_exactly_one_raw_conversion(
    stock_pqt_row: Callable[..., dict],
    hive_raw_tree: Callable[..., Path],
    tmp_path: Path,
    count_raw_conversions: Callable[[], int],
) -> None:
    """`StockDataset(cfg).from_raw_data().save()` must convert raw ONCE.

    Measured at 2 before the fix: the construction-time fallback materialises
    the whole panel and the caller asks for the same panel again microseconds
    later. Counted, not timed -- a timing assertion passes for the wrong
    reason on a fast machine.

    Reddening mutation: delete the early return from `from_raw_data()`.
    """
    config = _ingest_config(stock_pqt_row, hive_raw_tree, tmp_path)

    dataset = StockDataset(config).from_raw_data()
    dataset.save()

    assert count_raw_conversions() == 1
    assert Path(config.zarr_file_path).exists()
    assert xr.open_zarr(config.zarr_file_path).sizes["timestamp"] == 3


def test_the_constructors_panel_is_consumed_once_and_only_once(
    stock_pqt_row: Callable[..., dict],
    hive_raw_tree: Callable[..., Path],
    tmp_path: Path,
    count_raw_conversions: Callable[[], int],
) -> None:
    """The handoff is ONE-SHOT: a second `from_raw_data()` still re-converts.

    The fix removes one duplicate; it does not install a cache.

    Reddening mutation: do not clear the handoff at the top of
    `from_raw_data()` -- the second call then skips too and the count stays 1.
    """
    config = _ingest_config(stock_pqt_row, hive_raw_tree, tmp_path)

    dataset = StockDataset(config).from_raw_data()
    assert count_raw_conversions() == 1

    dataset.from_raw_data()

    assert count_raw_conversions() == 2


