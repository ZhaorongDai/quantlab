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
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import polars as pl
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
def ndx_anchor_html_fixture() -> str:
    """Synthetic HTML mimicking `https://stockanalysis.com/list/nasdaq-100-stocks/`,
    the Nasdaq-100 current-constituent anchor source: one table whose header is
    `No. | Symbol | Company Name | Market Cap | Stock Price | % Change | Revenue`.

    **102 body rows, not 100** -- that is the real row count observed against
    the live source (03.1-RESEARCH.md Finding 5). The Nasdaq-100 carries
    multiple share classes for some issuers (GOOGL/GOOG, FOX/FOXA), so an
    equality assertion against 100 fails against correct data.

    Named rows: `GOOGL`/`GOOG` and `FOX`/`FOXA` (the two share-class pairs
    that push the count past 100), and `NEWMEM` (added by an add-only
    change-log row and never removed, so it must be an anchor member for its
    interval to stay open-ended). The remaining 97 rows are generated
    `NDX001`..`NDX097` in a loop so the arithmetic 5 + 97 = 102 is visible in
    the source.

    **`LOGI` is deliberately ABSENT.** This table is a CURRENT-constituent
    snapshot, and the change log removes `LOGI` in 2018 -- so listing it here
    would assert "LOGI is a member today" and "LOGI stopped being a member in
    2018" simultaneously. That contradiction is precisely the third
    anchor/log disagreement `reconstruct_intervals()` now reconciles (CR-01):
    the anchor is authoritative for "is a member today", so an anchor row
    would correctly re-open LOGI's membership and it would no longer be the
    former member `test_nasdaq100_build_intervals_reconstructs_membership`
    asserts it is. `LOGI` still reaches the densified panel's symbol axis
    through its closed 2007-2018 interval -- that is the survivorship-bias
    guarantee, and it does not require an anchor row.

    The table carries NO `date_added` column -- neither real anchor source
    does -- which is what forces `Nasdaq100MembershipFetcher.fetch_anchor()`
    to synthesise an explicit all-null one.
    """
    named = ["GOOGL", "GOOG", "FOX", "FOXA", "NEWMEM"]
    generated = [f"NDX{i:03d}" for i in range(1, 98)]
    symbols = named + generated
    assert len(symbols) == 102, "anchor fixture must mirror the real 102-row shape"

    body_rows = "\n".join(
        "<tr>"
        f"<td>{index}</td><td>{symbol}</td><td>{symbol} Inc.</td>"
        f"<td>1.00B</td><td>10.00</td><td>0.10%</td><td>500.00M</td>"
        "</tr>"
        for index, symbol in enumerate(symbols, start=1)
    )
    return f"""
<html><body>
<table>
<thead>
<tr>
<th>No.</th><th>Symbol</th><th>Company Name</th><th>Market Cap</th>
<th>Stock Price</th><th>% Change</th><th>Revenue</th>
</tr>
</thead>
<tbody>
{body_rows}
</tbody>
</table>
</body></html>
"""


@pytest.fixture
def ndx_changes_html_fixture() -> str:
    """Synthetic HTML mirroring the real
    `https://en.wikipedia.org/wiki/Historical_components_of_the_Nasdaq-100`
    page's two-level, **six**-column header (`Date` spanning two rows; `Added`
    and `Removed` each spanning `Ticker`/`Security`; `Reason` spanning two
    rows). One fewer column than the S&P 500 page, which also carries `Refs`.

    Unlike the S&P 500 page there is no `id="changes"` to select on, so
    `Nasdaq100MembershipFetcher._parse_changes_table()` takes the single table
    and lets the base class's required-column check reject anything else.

    Rows, deliberately in non-sorted order so the reconstruction's own
    `sort("effective_date")` is exercised:

    1. `March 2, 2018` -- `TEMP1` added / `LOGI` removed. Closes LOGI's
       interval, so LOGI is a former member with a real `end_date`.
    2. `February 1, 2007` -- `LOGI` added / `CMVT` removed. This is the REAL
       earliest row on the live page and is what makes
       `PIT_COVERAGE_START = "2007-02-01"` (RESEARCH Finding 2). `CMVT` is
       itself left-censored here.
    3. `January 3, 2011` -- `NEWMEM` added, blank removed cells. An **add-only**
       row (18 of the live page's 226 rows are add-only, RESEARCH Finding 3).
    4. `June 15, 2015` -- blank added cells, `GONE1` removed. A **drop-only**
       row AND left-censored: `GONE1` is never added anywhere in this fixture,
       so it must start at `PIT_COVERAGE_START` with a WARNING.
    """
    rows = [
        ("March 2, 2018", "TEMP1", "Temp One Inc.", "LOGI", "Logitech", "Annual reconstitution"),
        ("February 1, 2007", "LOGI", "Logitech", "CMVT", "Comverse Technology", "Minimum weighting"),
        ("January 3, 2011", "NEWMEM", "New Member Corp.", "", "", "Annual reconstitution"),
        ("June 15, 2015", "", "", "GONE1", "Gone One Inc.", "Acquisition"),
    ]
    body_rows = "\n".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"""
<html><body>
<table>
<thead>
<tr>
<th rowspan="2">Date</th>
<th colspan="2">Added</th>
<th colspan="2">Removed</th>
<th rowspan="2">Reason</th>
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
    ndx_anchor_html_fixture: str,
    ndx_changes_html_fixture: str,
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
    - `Nasdaq100MembershipFetcher.ANCHOR_URL` -> `ndx_anchor_html_fixture`.
    - `Nasdaq100MembershipFetcher.CHANGES_URL` -> `ndx_changes_html_fixture`.

    No test in this suite makes a real network call.
    """
    from acquisition.universe import (
        Nasdaq100MembershipFetcher,
        NasdaqUniverseFetcher,
        SP500MembershipFetcher,
    )

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
        if url == Nasdaq100MembershipFetcher.ANCHOR_URL:
            return FakeResponse(text=ndx_anchor_html_fixture)
        if url == Nasdaq100MembershipFetcher.CHANGES_URL:
            return FakeResponse(text=ndx_changes_html_fixture)
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
    `base/data.py:BaseDataset.config`'s setter calls `_reset_symbols()` (which
    calls `read()`) whenever `DatasetConfig.symbols` is not None, so the file
    must already exist by the time a caller hands the config to a
    `MarketDataset`.
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


_STOCK_PQT_COLUMNS = [
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


@pytest.fixture
def stock_pqt_row() -> Callable[..., dict]:
    """Return a row-builder callable `(date_str, symbol, close=100.0) -> dict`.

    Mirrors the exact field set/dtypes `acquisition/tiingo.py:TiingoAcquisition`
    writes (Tiingo's EOD columns + divCash/splitFactor + renamed
    timestamp/symbol) so synthetic fixtures schema-match real vendor output
    when `pl.concat()`'d together in `StockDataset._raw_data_to_xr()`.

    Promoted out of `tests/test_stock_dataset.py`'s private `_row()` helper
    (03-VALIDATION.md Wave-0 gap) so Phase-3 stock factor tests reuse it
    instead of duplicating it a third time.
    """

    def _row(date_str: str, symbol: str, close: float = 100.0) -> dict:
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

    return _row


@pytest.fixture
def write_stock_pqt() -> Callable[..., Path]:
    """Factory fixture. Call as `write_stock_pqt(path, rows)` to write a raw
    Tiingo-shaped parquet file (creating parent directories as needed) and get
    the written `Path` back.

    Promoted out of `tests/test_stock_dataset.py`'s private
    `_write_stock_pqt()` helper (03-VALIDATION.md Wave-0 gap).
    """

    def _write(path: Path, rows: list[dict]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pl.DataFrame(rows).select(_STOCK_PQT_COLUMNS)
        df.write_parquet(path)
        return path

    return _write


@pytest.fixture
def stock_zarr(tmp_path: Path) -> Callable[..., DatasetConfig]:
    """Factory fixture. Call as
    `stock_zarr(symbols=None, periods=60, seed=0)` to write a synthetic
    Tiingo-shaped Zarr store at `{tmp_path}/stock/stock.zarr` and get back a
    `DatasetConfig` pointing at it.

    Emits BOTH the raw lowercase group (`open`/`high`/`low`/`close`/`volume`)
    AND the adjusted group (`adjOpen`/`adjHigh`/`adjLow`/`adjClose`/
    `adjVolume`): `dataset/stock.py:StockDataset._to_kunquant()` drops the
    former and renames the latter onto those names, so a store missing either
    group raises.

    Same seeded, strictly-positive random-walk generator as
    `spot_kline_zarr`, and the same write-before-config ordering rule.
    """

    def _build(
        symbols: Optional[list[str]] = None,
        periods: int = 60,
        seed: int = 0,
    ) -> DatasetConfig:
        symbol_list = list(symbols) if symbols is not None else ["AAPL", "MSFT"]
        timestamps = pd.date_range("2024-01-01", periods=periods, freq="D")
        rng = np.random.default_rng(seed)

        base = _positive_random_walk(rng, periods, len(symbol_list))
        volume = _positive_random_walk(rng, periods, len(symbol_list)) * 10.0

        variables = {
            "open": base * 0.99,
            "high": base * 1.02,
            "low": base * 0.98,
            "close": base,
            "volume": volume,
            "adjOpen": base * 0.99,
            "adjHigh": base * 1.02,
            "adjLow": base * 0.98,
            "adjClose": base,
            "adjVolume": volume,
        }

        dataset = xr.Dataset(
            {
                name: (["timestamp", "symbol"], values)
                for name, values in variables.items()
            },
            coords={"timestamp": timestamps, "symbol": symbol_list},
        )

        stock_dir = tmp_path / "stock"
        stock_dir.mkdir(parents=True, exist_ok=True)
        zarr_path = stock_dir / "stock.zarr"
        dataset.to_zarr(zarr_path, mode="w")

        return DatasetConfig(
            raw_data_dir_path=str(stock_dir / "raw"),
            zarr_file_path=str(zarr_path),
            catalog_path=str(stock_dir / "catalog"),
            market="us_equity",
            frequency="1d",
        )

    return _build
