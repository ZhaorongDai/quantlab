"""Shared pytest fixtures for the quantlab test suite.

These fixtures provide synthetic market-data inputs (Binance CSV rows, Tiingo
JSON responses) and a mocked Tiingo client so downstream tests never need a
real network call or a real API credential.

IMPORTANT: this module must have zero import-time dependency on
`acquisition.tiingo` (it does not exist yet as of Phase 2 Wave 1) so that
`pytest --collect-only` succeeds today regardless of which feature plans have
landed. `mock_tiingo_client` only references it as a dotted string inside
`monkeypatch.setattr(...)`, evaluated lazily when the fixture is used by a
test, never at module import time.
"""

from pathlib import Path
from typing import Callable, Optional

import pytest


@pytest.fixture
def binance_csv_rows() -> list[list]:
    """Return a small list of raw row-tuples matching the column order of
    `enums.data.BinanceCSVHeaders.SPOT` (12 columns), with `Open time` as
    millisecond epoch ints.
    """
    return [
        [
            1704067200000,  # Open time
            "42000.00",  # Open
            "42500.00",  # High
            "41800.00",  # Low
            "42300.00",  # Close
            "123.456",  # Volume
            1704153599999,  # Close time
            "5234567.89",  # Quote asset volume
            1000,  # Number of trades
            "60.123",  # Taker buy base asset volume
            "2534567.89",  # Taker buy quote asset volume
            "0",  # Ignore
        ],
        [
            1704153600000,
            "42300.00",
            "43000.00",
            "42100.00",
            "42800.00",
            "150.789",
            1704239999999,
            "6412345.67",
            1200,
            "80.456",
            "3412345.67",
            "0",
        ],
    ]


@pytest.fixture
def write_binance_csv(
    tmp_path: Path, binance_csv_rows: list[list]
) -> Callable[..., Path]:
    """Factory fixture. Call as
    `write_binance_csv(symbol="BTCUSDT", year_month="2024-01", rows=None)` to
    write a headerless CSV file at `{tmp_path}/{symbol}-1d-{year_month}.csv`
    using `binance_csv_rows()` as the default row data if `rows` is not given.
    Returns the written `Path`.
    """

    def _write(
        symbol: str = "BTCUSDT",
        year_month: str = "2024-01",
        rows: Optional[list[list]] = None,
    ) -> Path:
        rows_to_write = rows if rows is not None else binance_csv_rows
        file_path = tmp_path / f"{symbol}-1d-{year_month}.csv"
        with open(file_path, "w") as f:
            for row in rows_to_write:
                f.write(",".join(str(value) for value in row) + "\n")
        return file_path

    return _write


@pytest.fixture
def tiingo_json_response() -> list[dict]:
    """Return a small list of dicts matching the field set
    `open,high,low,close,volume,adjOpen,adjHigh,adjLow,adjClose,adjVolume,
    divCash,splitFactor,date` with `date` as ISO-8601 strings.
    """
    return [
        {
            "date": "2024-01-02T00:00:00.000Z",
            "open": 100.0,
            "high": 102.5,
            "low": 99.5,
            "close": 101.0,
            "volume": 1000000,
            "adjOpen": 100.0,
            "adjHigh": 102.5,
            "adjLow": 99.5,
            "adjClose": 101.0,
            "adjVolume": 1000000,
            "divCash": 0.0,
            "splitFactor": 1.0,
        },
        {
            "date": "2024-01-03T00:00:00.000Z",
            "open": 101.0,
            "high": 103.0,
            "low": 100.5,
            "close": 102.5,
            "volume": 1100000,
            "adjOpen": 101.0,
            "adjHigh": 103.0,
            "adjLow": 100.5,
            "adjClose": 102.5,
            "adjVolume": 1100000,
            "divCash": 0.0,
            "splitFactor": 1.0,
        },
    ]


@pytest.fixture
def mock_tiingo_client(monkeypatch, tiingo_json_response: list[dict]) -> type:
    """Return a `FakeTiingoClient` class and, as a side effect, patch
    `acquisition.tiingo.TiingoClient` to it so `TiingoAcquisition` (once it
    exists) never makes a real network call. Also sets `TIINGO_API_KEY` to a
    fake value so no real credential is ever required by tests.

    Records every `get_ticker_price(...)` call's kwargs on
    `FakeTiingoClient.calls` (a list) so tests can assert on
    `startDate`/`columns`/`frequency` arguments passed by `TiingoAcquisition`.
    """

    class FakeTiingoClient:
        calls: list[dict] = []

        def __init__(self, *args, **kwargs) -> None:
            pass

        def get_ticker_price(self, ticker, **kwargs):
            FakeTiingoClient.calls.append({"ticker": ticker, **kwargs})
            return tiingo_json_response

    # Reset call log per-test so tests don't leak state across each other.
    FakeTiingoClient.calls = []

    monkeypatch.setattr(
        "acquisition.tiingo.TiingoClient", FakeTiingoClient, raising=False
    )
    monkeypatch.setenv("TIINGO_API_KEY", "test-key-not-real")

    return FakeTiingoClient
