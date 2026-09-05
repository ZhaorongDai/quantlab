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

import io
import zipfile
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from base.config import DatasetConfig


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


@pytest.fixture
def sp500_anchor_csv_rows() -> str:
    """CSV text mimicking datasets/s-and-p-500-companies's `constituents.csv`
    real columns. Includes AAPL (long-standing, early `Date added`), MSFT,
    TSLA with `Date added = "2020-12-21"` (Tesla's real, publicly documented
    S&P 500 inclusion date), and `ADDED1` (matches the `sp500_changes_html_
    fixture`'s earliest, 1976-07-01-dated row so it round-trips as a still-
    current, open-ended member).
    """
    header = (
        "Symbol,Security,GICS Sector,GICS Sub-Industry,"
        "Headquarters Location,Date added,CIK,Founded"
    )
    rows = [
        header,
        'AAPL,Apple Inc.,Information Technology,Technology Hardware,'
        '"Cupertino, California",1982-11-30,0000320193,1976',
        'MSFT,Microsoft Corp.,Information Technology,Systems Software,'
        '"Redmond, Washington",1994-06-01,0000789019,1975',
        'TSLA,Tesla Inc.,Consumer Discretionary,Automobile Manufacturers,'
        '"Austin, Texas",2020-12-21,0001318605,2003',
        'ADDED1,Added1 Co,Industrials,Diversified Industrials,'
        '"New York, New York",1976-07-01,0000000001,1970',
    ]
    return "\n".join(rows) + "\n"


@pytest.fixture
def sp500_changes_html_fixture() -> str:
    """Synthetic HTML with `<table id="changes">` mirroring the real
    Wikipedia "Historical components of the S&P 500" page's 2-level-header,
    7-column shape (Effective Date | Added Ticker | Added Security |
    Removed Ticker | Removed Security | Reason | Refs).

    Rows, in original (non-sorted) order:
    1. July 1, 1976 -- aligns with `SP500MembershipFetcher.PIT_COVERAGE_
       START`; `ADDED1` added (never removed -- matches the anchor CSV).
    2. January 1, 1980 -- `REENTRY` added (first membership span).
    3. January 1, 1985 -- `ZZZZ` removed with NO matching prior "added" row
       anywhere in this fixture (exercises the left-censored path).
    4. January 1, 1990 -- `REENTRY` removed (closes the first span).
    5. January 1, 2000 -- `REENTRY` re-added (second membership span,
       exercises the re-entry path).
    6. December 21, 2020 -- `TSLA` added / `AIV` removed. Tesla's real,
       publicly documented S&P 500 addition date.
    """
    rows = [
        ("July 1, 1976", "ADDED1", "Added1 Co", "", "", "Initial", ""),
        ("January 1, 1980", "REENTRY", "Reentry Co", "", "", "Initial addition", ""),
        ("January 1, 1985", "", "", "ZZZZ", "ZZZZ Co", "Removed (left-censored)", ""),
        ("January 1, 1990", "", "", "REENTRY", "Reentry Co", "Removed", ""),
        ("January 1, 2000", "REENTRY", "Reentry Co", "", "", "Re-added", ""),
        (
            "December 21, 2020",
            "TSLA",
            "Tesla, Inc.",
            "AIV",
            "Apartment Investment & Management Co",
            "Market cap change",
            "",
        ),
    ]
    body_rows = "\n".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"""
<html><body>
<table id="changes">
<thead>
<tr>
<th rowspan="2">Effective Date</th>
<th colspan="2">Added</th>
<th colspan="2">Removed</th>
<th rowspan="2">Reason</th>
<th rowspan="2">Refs</th>
</tr>
<tr>
<th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th>
</tr>
</thead>
<tbody>
{body_rows}
</tbody>
</table>
</body></html>
"""


@pytest.fixture
def mock_universe_fetchers(
    monkeypatch,
    sp500_anchor_csv_rows: str,
    sp500_changes_html_fixture: str,
) -> Callable[..., object]:
    """Patch `acquisition.universe.requests.get` to return a `FakeResponse`
    (with `.status_code`, `.text`, `.content`, `.raise_for_status()`) keyed
    by requested URL, matching each fetcher's real class-constant URL:

    - `NasdaqUniverseFetcher.SOURCE_URL` -> an in-memory zip wrapping a
      synthetic `supported_tickers.csv` with rows spanning NASDAQ/NYSE,
      Stock/ETF, USD/EUR (and one NASDAQ/Stock/USD row with a real
      `endDate` to exercise the delisted-exclusion path).
    - `SP500MembershipFetcher.ANCHOR_URL` -> `sp500_anchor_csv_rows`.
    - `SP500MembershipFetcher.CHANGES_URL` -> `sp500_changes_html_fixture`.

    No test in this suite makes a real network call.
    """
    from acquisition.universe import NasdaqUniverseFetcher, SP500MembershipFetcher

    nasdaq_csv = (
        "ticker,exchange,assetType,priceCurrency,startDate,endDate\n"
        "AAPL,NASDAQ,Stock,USD,1980-12-12,\n"
        "MSFT,NASDAQ,Stock,USD,1986-03-13,\n"
        "NYSE1,NYSE,Stock,USD,1990-01-01,\n"
        "ETF1,NASDAQ,ETF,USD,2000-01-01,\n"
        "EURO1,NASDAQ,Stock,EUR,2000-01-01,\n"
        "DELISTED1,NASDAQ,Stock,USD,1990-01-01,2020-01-01\n"
    )
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("supported_tickers.csv", nasdaq_csv)
    zip_bytes = zip_buffer.getvalue()

    class FakeResponse:
        def __init__(self, content: bytes = b"", text: str = ""):
            self.status_code = 200
            self.content = content
            self.text = text

        def raise_for_status(self) -> None:
            pass

    def fake_get(url, *args, **kwargs):
        if url == NasdaqUniverseFetcher.SOURCE_URL:
            return FakeResponse(content=zip_bytes)
        if url == SP500MembershipFetcher.ANCHOR_URL:
            return FakeResponse(text=sp500_anchor_csv_rows)
        if url == SP500MembershipFetcher.CHANGES_URL:
            return FakeResponse(text=sp500_changes_html_fixture)
        raise AssertionError(f"Unexpected URL requested in test: {url}")

    monkeypatch.setattr("acquisition.universe.requests.get", fake_get)
    return fake_get


# ---------------------------------------------------------------------------
# Phase 3 synthetic-Zarr fixtures (03-01-PLAN.md, FACTOR-01)
#
# Factor computation needs a much larger input than Phase 2's
# `write_binance_csv` fixture provides (2 rows / 1 symbol): rolling factor
# windows need ~60 timestamps and the streaming replay needs 8 symbols. These
# factories synthesise a seeded, strictly-positive market-data panel and write
# it straight to a Zarr store, skipping raw CSV/parquet ingestion entirely.
#
# Import-safety rule (see module docstring): these fixtures import only
# numpy/pandas/xarray and `base.config` -- never a `dataset.*` or `factor.*`
# module -- so `pytest --collect-only` stays green.
# ---------------------------------------------------------------------------


def _positive_random_walk(
    rng: np.random.Generator, periods: int, num_symbols: int
) -> np.ndarray:
    """Return a `[periods, num_symbols]` float array that is strictly greater
    than 1.0 everywhere.

    KunQuant factor graphs divide by price and volume columns, so a zero or
    negative entry would poison every downstream factor with inf/NaN. Starting
    at 100.0, cumulatively summing normal increments and then taking
    `abs(...) + 1.0` guarantees positivity without destroying the
    walk's time-series structure.
    """
    increments = rng.normal(loc=0.0, scale=1.0, size=(periods, num_symbols))
    walk = 100.0 + np.cumsum(increments, axis=0)
    return np.abs(walk) + 1.0


@pytest.fixture
def spot_kline_zarr(tmp_path: Path) -> Callable[..., DatasetConfig]:
    """Factory fixture. Call as
    `spot_kline_zarr(symbols=None, periods=60, seed=0)` to write a synthetic
    Binance-shaped Zarr store at `{tmp_path}/spot/klines.zarr` and get back a
    `DatasetConfig` pointing at it.

    Variables are the raw Binance Title-Case names `SpotKlineDataset` persists
    (`Open`, `High`, `Low`, `Close`, `Volume`, `Quote asset volume`) over dims
    `["timestamp", "symbol"]` -- `SpotKlineDataset._to_kunquant()` renames them
    to KunQuant's lowercase `open/high/low/close/volume/amount` at the
    boundary.

    The store is written to disk BEFORE the `DatasetConfig` is constructed:
    `base/data.py:Dataset.config`'s setter calls `_reset_symbols()` (which
    calls `read()`) whenever `DatasetConfig.symbols` is not None, so the file
    must already exist by the time a caller hands the config to a `Dataset`.
    """

    def _build(
        symbols: Optional[list[str]] = None,
        periods: int = 60,
        seed: int = 0,
    ) -> DatasetConfig:
        symbol_list = (
            list(symbols)
            if symbols is not None
            else [f"S{i}USDT" for i in range(8)]
        )
        timestamps = pd.date_range("2024-01-01", periods=periods, freq="D")
        rng = np.random.default_rng(seed)

        base = _positive_random_walk(rng, periods, len(symbol_list))
        volume = _positive_random_walk(rng, periods, len(symbol_list)) * 10.0

        variables = {
            "Open": base * 0.99,
            "High": base * 1.02,
            "Low": base * 0.98,
            "Close": base,
            "Volume": volume,
            "Quote asset volume": volume * base,
        }

        dataset = xr.Dataset(
            {
                name: (["timestamp", "symbol"], values)
                for name, values in variables.items()
            },
            coords={"timestamp": timestamps, "symbol": symbol_list},
        )

        spot_dir = tmp_path / "spot"
        spot_dir.mkdir(parents=True, exist_ok=True)
        zarr_path = spot_dir / "klines.zarr"
        dataset.to_zarr(zarr_path, mode="w")

        return DatasetConfig(
            raw_data_dir_path=str(spot_dir / "raw"),
            zarr_file_path=str(zarr_path),
            catalog_path=str(spot_dir / "catalog"),
            market="crypto_spot",
            frequency="1d",
        )

    return _build
