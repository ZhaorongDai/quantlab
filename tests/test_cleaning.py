"""Unit tests for dataset/cleaning.py — the shared raw-market-data cleaning
module (Phase 2 Plan 03, CONTEXT.md D-05..D-08).

Tests 1-3 cover dedup_raw_frame() (D-05, Task 1).
"""

import polars as pl
import pytest

from dataset.cleaning import dedup_raw_frame


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
