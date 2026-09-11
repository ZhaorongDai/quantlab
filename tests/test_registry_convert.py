"""`quantlab.acquisition.registry.convert()` -- the registry-level raw->Zarr
entry point (DATA-07, phase 03.5).

The requirement these tests exist for is stated as a CALL SITE, not as a
behaviour: an in-process caller holding only a `SourceDescriptor` and a
`DatasetConfig` converts an already-persisted raw parquet tier into a Zarr
store, without naming `StockDataset`, `TiingoAcquisition` or any `ingest_*.py`
anywhere. Every test below therefore reaches the conversion through
`convert(DESCRIPTOR, config)` and never through the Dataset class -- a test
that constructed the dataset itself would pass while the requirement failed.

Offline, all of it. No credential is read, no network call is made, and every
path is under `tmp_path`. The raw tier is a hive-partitioned parquet tree
written by the shared `hive_raw_tree` fixture, in the shape
`StockDataset._scan_raw` asserts: `Path(raw_data_dir_path).name == vendor`.
"""

from pathlib import Path
from typing import Callable

import pytest
import xarray as xr

from quantlab.base.config import DatasetConfig
from quantlab.base.data import ConversionResult
from quantlab.acquisition.registry import DataSourceRegistry, convert

#: Two calendar years, a handful of observed days in each, so a `year` window
#: is cheap and there are exactly TWO planned windows to write, skip and count.
#: Sparse across symbols on purpose (`B` trades only in 2024): a window that
#: derived its own symbol axis would come back with one column instead of the
#: pinned two, which is the misalignment `from_raw_data_chunked`'s pinned axis
#: exists to prevent -- and which `pinned_symbols` in the result reports.
_YEARS = (2023, 2024)
_DAYS_PER_YEAR = ("01-04", "06-15", "12-28")
_EXPECTED_WINDOWS = len(_YEARS)
_EXPECTED_SYMBOLS = 2


def _raw_rows(stock_pqt_row: Callable[..., dict]) -> list[dict]:
    rows = []
    for year in _YEARS:
        for day in _DAYS_PER_YEAR:
            date_str = f"{year}-{day}"
            rows.append(stock_pqt_row(date_str, "A", close=100.0))
            if year == 2024:
                rows.append(stock_pqt_row(date_str, "B", close=100.0))
    return rows


@pytest.fixture
def tiingo_raw_tier(
    stock_pqt_row: Callable[..., dict],
    hive_raw_tree: Callable[..., Path],
    tmp_path: Path,
) -> Callable[..., DatasetConfig]:
    """Factory. `tiingo_raw_tier()` writes the two-year raw parquet panel once
    and returns a `DatasetConfig` a caller could hand straight to `convert()`.

    Calling it twice returns configs pointing at the SAME raw tree and the SAME
    Zarr path, which is what makes the idempotency probe a genuine second run
    over the first run's store rather than a fresh one.
    """
    raw_dir = tmp_path / "raw"
    hive_raw_tree(raw_dir, "tiingo", _raw_rows(stock_pqt_row), batch_key="panel")

    def _build(store_name: str = "out.zarr") -> DatasetConfig:
        return DatasetConfig(
            raw_data_dir_path=str(raw_dir / "tiingo"),
            zarr_file_path=str(tmp_path / store_name),
            catalog_path=str(tmp_path / "catalog"),
            market="us_equity",
            frequency="1d",
            vendor="tiingo",
        )

    return _build


def test_convert_writes_a_real_zarr_store_and_returns_a_conversion_result(
    tiingo_raw_tier: Callable[..., DatasetConfig],
) -> None:
    """DATA-07's tracer: descriptor + config in, Zarr store on disk out.

    The assertions are deliberately split between the STORE and the RESULT.
    Asserting only the returned object would pass against a `convert()` that
    computed counts and wrote nothing; asserting only the store would pass
    against one that wrote but reported nothing a caller could render (D-04).
    Both halves are the requirement.
    """
    descriptor = DataSourceRegistry.get("tiingo")
    config = tiingo_raw_tier()

    result = convert(descriptor, config)

    # -- the store is real --------------------------------------------------
    assert Path(config.zarr_file_path).exists()
    stored = xr.open_zarr(config.zarr_file_path)
    assert set(stored.dims) >= {"timestamp", "symbol"}
    assert stored.sizes["symbol"] == _EXPECTED_SYMBOLS
    assert stored.sizes["timestamp"] == len(_YEARS) * len(_DAYS_PER_YEAR)

    # -- the outcome is renderable without reading that store ---------------
    assert isinstance(result, ConversionResult)
    assert result.zarr_path == config.zarr_file_path
    assert Path(result.ledger_path).exists()
    assert result.granularity == "year"
    assert result.pinned_symbols == _EXPECTED_SYMBOLS
    assert result.windows_planned == _EXPECTED_WINDOWS
    assert result.windows_written == _EXPECTED_WINDOWS
    assert result.windows_skipped == 0
    assert result.rows_written == len(_YEARS) * len(_DAYS_PER_YEAR)
    assert result.peak_window_bytes is not None and result.peak_window_bytes > 0
    assert result.resumed is False
    assert result.cancelled is False
    # D-11: `convert()` runs no guard and invents no estimate. `None` here is
    # the honest answer when the caller offered none, not a missing value.
    assert result.predicted_peak_bytes is None


def test_a_second_convert_over_the_same_window_writes_nothing_twice(
    tiingo_raw_tier: Callable[..., DatasetConfig],
) -> None:
    """DATA-07's idempotency probe row, as an assertion rather than as prose.

    A second call over the same config must not re-append a single window --
    re-appending would duplicate every timestamp in the store, which is the
    silent corruption the chunk ledger exists to make impossible. The proof is
    that the second run's SKIPPED count equals the first run's WRITTEN count
    and its own written count is zero.
    """
    descriptor = DataSourceRegistry.get("tiingo")
    first = convert(descriptor, tiingo_raw_tier())

    second = convert(descriptor, tiingo_raw_tier())

    assert second.windows_written == 0
    assert second.windows_skipped == first.windows_written
    assert second.resumed is True
    assert second.rows_written == 0
    # Nothing was materialised, so there is no observed peak to report.
    assert second.peak_window_bytes is None

    # The store itself is unchanged -- the counts above would also be produced
    # by a run that skipped the ledger check and appended anyway if the ledger
    # were the only thing consulted.
    stored = xr.open_zarr(first.zarr_path)
    assert stored.sizes["timestamp"] == len(_YEARS) * len(_DAYS_PER_YEAR)
    assert stored.sizes["symbol"] == _EXPECTED_SYMBOLS


def test_predicted_peak_is_echoed_back_untouched(
    tiingo_raw_tier: Callable[..., DatasetConfig],
) -> None:
    """`predicted_peak_bytes` is the CALLER'S number, carried not computed.

    D-11 leaves the RAM guard at the call site, so the entry point has no
    roster category to size against. Echoing the caller's own estimate into
    the result is what lets a report put prediction beside outcome without
    `convert()` pretending to be self-protecting.
    """
    result = convert(
        DataSourceRegistry.get("tiingo"),
        tiingo_raw_tier(),
        predicted_peak_bytes=123_456,
    )

    assert result.predicted_peak_bytes == 123_456
    # The echo must not disturb anything that was measured.
    assert result.windows_written == _EXPECTED_WINDOWS


def test_run_still_returns_an_acquisition_result_and_converts_nothing() -> None:
    """SC-2's surviving first half: conversion is a SEPARATE function.

    Pinned structurally rather than by running an acquisition (which would
    need a credential): `run` and `convert` are two module-level functions
    with two different return annotations, and `run` grew no conversion
    keyword. A `mode=`/`to_zarr=` flag appearing on `run` is exactly the
    regression 03.4 D-14 forbids, and it would be invisible to every
    behavioural test in this file.
    """
    import inspect

    from quantlab.acquisition import registry
    from quantlab.base.acquisition import AcquisitionResult

    assert registry.run.__annotations__["return"] is AcquisitionResult
    assert registry.convert.__annotations__["return"] is ConversionResult

    run_params = set(inspect.signature(registry.run).parameters)
    assert not run_params & {"mode", "to_zarr", "convert", "granularity"}
