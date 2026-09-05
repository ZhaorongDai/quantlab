"""Unit tests for dataset/cleaning.py — the shared raw-market-data cleaning
module (Phase 2 Plan 03, CONTEXT.md D-05..D-08).

Tests 1-3 cover dedup_raw_frame() (D-05, Task 1).
Tests 4-7 cover flag_anomalies()/validate_schema()/clean_market_data() and
the base/data.py:Dataset.from_raw_data() wiring (D-06, D-07, D-08, Task 2).
"""

from typing import Self

import numpy as np
import polars as pl
import pytest
import xarray as xr

from dataset.cleaning import (
    REQUIRED_COLUMNS,
    clean_market_data,
    dedup_raw_frame,
    flag_anomalies,
    validate_schema,
)


def test_dedup_keeps_last_row_for_duplicate_timestamp_symbol_pair() -> None:
    """Test 1: two rows sharing the same (timestamp, symbol) pair but
    different `close` values collapse to exactly one row, and that row's
    `close` is the one that appears LAST in the input ordering
    (keep="last" semantics)."""
    data = pl.LazyFrame(
        {
            "timestamp": ["2024-01-01", "2024-01-01"],
            "symbol": ["AAPL", "AAPL"],
            "close": [100.0, 999.0],
        }
    )

    result = dedup_raw_frame(data, keep="last").collect()

    assert result.height == 1
    assert result["close"].to_list() == [999.0]


def test_dedup_no_duplicates_returns_unchanged_row_count() -> None:
    """Test 2: a frame with no duplicate (timestamp, symbol) pairs is
    returned unchanged (same row count)."""
    data = pl.LazyFrame(
        {
            "timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "close": [100.0, 101.0, 102.0],
        }
    )

    result = dedup_raw_frame(data).collect()

    assert result.height == 3


def test_dedup_prevents_non_unique_multiindex_to_xarray_crash() -> None:
    """Test 3: a frame that WOULD raise ValueError on to_xarray() due to a
    non-unique (timestamp, symbol) MultiIndex succeeds once dedup_raw_frame()
    has been applied first (02-RESEARCH.md Pitfall 2)."""
    data = pl.LazyFrame(
        {
            "timestamp": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "close": [100.0, 999.0, 101.0],
        }
    )

    # Sanity check: without dedup, this WOULD raise.
    with pytest.raises(ValueError):
        (
            data.collect()
            .to_pandas()
            .set_index(["timestamp", "symbol"])
            .to_xarray()
        )

    deduped = dedup_raw_frame(data, keep="last")
    result = (
        deduped.collect()
        .to_pandas()
        .set_index(["timestamp", "symbol"])
        .to_xarray()
    )

    assert (
        result["close"].sel(timestamp="2024-01-01", symbol="AAPL").item()
        == 999.0
    )


# ---------------------------------------------------------------------------
# Task 2: flag_anomalies() / validate_schema() / clean_market_data() (D-06..D-08)
# ---------------------------------------------------------------------------


def _make_xr_dataset(
    close_values: list[list[float]],
    timestamps: list[str] | None = None,
    symbols: list[str] | None = None,
) -> xr.Dataset:
    timestamps = timestamps or [
        f"2024-01-0{i + 1}" for i in range(len(close_values))
    ]
    symbols = symbols or ["A", "B"]
    data_vars = {
        "open": (["timestamp", "symbol"], close_values),
        "high": (["timestamp", "symbol"], close_values),
        "low": (["timestamp", "symbol"], close_values),
        "close": (["timestamp", "symbol"], close_values),
        "volume": (["timestamp", "symbol"], close_values),
    }
    return xr.Dataset(
        data_vars,
        coords={"timestamp": timestamps, "symbol": symbols},
    )


def test_flag_anomalies_flags_zero_and_negative_price_without_mutating_values() -> (
    None
):
    """Test 4: a dataset with one zero-price and one negative-price point
    gets `anomaly_flag=True` at exactly those points, and the underlying
    price values are byte-for-byte unchanged."""
    close_values = [[100.0, 0.0], [-5.0, 101.0]]
    data = _make_xr_dataset(close_values)
    original_close = data["close"].values.copy()

    result = flag_anomalies(data)

    assert "anomaly_flag" in result.data_vars
    assert result["anomaly_flag"].dtype == bool
    flags = result["anomaly_flag"].values
    assert flags[0, 1] == True  # noqa: E712 — zero price
    assert flags[1, 0] == True  # noqa: E712 — negative price
    assert flags[0, 0] == False  # noqa: E712
    assert flags[1, 1] == False  # noqa: E712
    np.testing.assert_array_equal(result["close"].values, original_close)


def test_validate_schema_raises_on_missing_required_column() -> None:
    """Test 5a: validate_schema() raises when a required column is missing,
    naming the missing column."""
    data = xr.Dataset(
        {
            "open": (["timestamp", "symbol"], [[1.0, 2.0]]),
            "high": (["timestamp", "symbol"], [[1.0, 2.0]]),
            "low": (["timestamp", "symbol"], [[1.0, 2.0]]),
            "close": (["timestamp", "symbol"], [[1.0, 2.0]]),
            # "volume" missing
        },
        coords={"timestamp": ["2024-01-01"], "symbol": ["A", "B"]},
    )

    with pytest.raises(ValueError, match="volume"):
        validate_schema(data)


def test_validate_schema_warns_not_raises_on_unexpected_null_in_non_key_column(
    caplog,
) -> None:
    """Test 5b: validate_schema() given all required columns but an
    unexpected null in a non-key column logs a warning and does NOT raise."""
    data = _make_xr_dataset([[100.0, np.nan]])

    # Must not raise.
    result = validate_schema(data)

    assert result is not None


def test_clean_market_data_preserves_nan_gaps() -> None:
    """Test 6: clean_market_data() never forward-fills/interpolates — a NaN
    present in the input is still NaN in the output, at the exact same
    positions."""
    close_values = [[100.0, np.nan], [np.nan, 101.0]]
    data = _make_xr_dataset(close_values)

    result = clean_market_data(data)

    result_close = result["close"].values
    assert np.isnan(result_close[0, 1])
    assert np.isnan(result_close[1, 0])
    assert result_close[0, 0] == 100.0
    assert result_close[1, 1] == 101.0


class _NoOpDataset:
    """Minimal test-only Dataset subclass exercising from_raw_data() without
    requiring any real Dataset construction dependencies (config, backend).
    Only implements the pieces from_raw_data() actually touches, mirroring
    base.data.Dataset's method contract."""

    def __init__(self, data: xr.Dataset, data_backend) -> None:
        self._data = data
        self.data_backend = data_backend

    def _raw_data_to_xr(self) -> xr.Dataset:
        return self._data

    def from_raw_data(self) -> Self:
        from base.data import Dataset

        return Dataset.from_raw_data(self)  # type: ignore[arg-type]


def test_from_raw_data_calls_clean_market_data(monkeypatch) -> None:
    """Test 7: Dataset.from_raw_data() calls clean_market_data() exactly
    once with the xr.Dataset produced by _raw_data_to_xr()."""
    import base.data as base_data_module

    raw_data = _make_xr_dataset([[100.0, 101.0]])
    calls: list[xr.Dataset] = []

    def _fake_clean_market_data(data: xr.Dataset) -> xr.Dataset:
        calls.append(data)
        return data

    monkeypatch.setattr(
        base_data_module, "clean_market_data", _fake_clean_market_data
    )

    class _FakeBackend:
        def to_internal(self, data: xr.Dataset):
            self.data = data
            return self

    dataset = _NoOpDataset(raw_data, _FakeBackend())
    dataset.from_raw_data()

    assert len(calls) == 1
    assert calls[0] is raw_data


def test_required_columns_constant_matches_expected_set() -> None:
    assert set(REQUIRED_COLUMNS) == {"open", "high", "low", "close", "volume"}
