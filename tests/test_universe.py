"""Tests for acquisition/universe.py -- point-in-time correctness,
left-censoring, re-entry, and graceful degradation.

No test in this module makes a real network call: all `requests.get` calls
are patched by the `mock_universe_fetchers` fixture in `tests/conftest.py`.
"""

import polars as pl
import pytest
from loguru import logger

from acquisition.universe import (
    Nasdaq100MembershipFetcher,
    NasdaqUniverseFetcher,
    SP500MembershipFetcher,
    UniverseCatalog,
)
from base.config import UniverseConfig


def _make_config(tmp_path) -> UniverseConfig:
    return UniverseConfig(
        output_path=str(tmp_path / "reference" / "universe.parquet"),
        cache_dir=str(tmp_path / "reference" / "_cache"),
    )


def test_nasdaq_fetcher_filters_exchange_assettype_currency(mock_universe_fetchers):
    result = NasdaqUniverseFetcher().fetch()

    symbols = set(result["symbol"].to_list())
    assert symbols == {"AAPL", "MSFT", "DELISTED1"}
    assert "NYSE1" not in symbols  # wrong exchange
    assert "ETF1" not in symbols  # wrong assetType
    assert "EURO1" not in symbols  # wrong priceCurrency


def test_reconstruct_intervals_reentry(mock_universe_fetchers):
    fetcher = SP500MembershipFetcher(cache_dir="unused")
    anchor = fetcher.fetch_anchor()
    changes = fetcher.fetch_changes()

    intervals = fetcher.reconstruct_intervals(anchor, changes)
    reentry_rows = (
        intervals.filter(pl.col("symbol") == "REENTRY")
        .sort("start_date")
        .to_dicts()
    )

    assert len(reentry_rows) >= 2
    for earlier, later in zip(reentry_rows, reentry_rows[1:]):
        assert earlier["end_date"] is not None
        assert earlier["end_date"] <= later["start_date"]


def test_reconstruct_intervals_left_censored(mock_universe_fetchers):
    fetcher = SP500MembershipFetcher(cache_dir="unused")
    anchor = fetcher.fetch_anchor()
    changes = fetcher.fetch_changes()

    # loguru does not propagate to stdlib `logging` (pytest's `caplog`) by
    # default -- attach a temporary in-memory sink to assert on the warning
    # instead, per 02-08-PLAN.md's "via caplog or a loguru sink" allowance.
    captured_messages: list[str] = []
    sink_id = logger.add(captured_messages.append, level="WARNING", format="{message}")
    try:
        intervals = fetcher.reconstruct_intervals(anchor, changes)
    finally:
        logger.remove(sink_id)

    zzzz_rows = intervals.filter(pl.col("symbol") == "ZZZZ").to_dicts()
    assert len(zzzz_rows) == 1
    assert zzzz_rows[0]["start_date"] == SP500MembershipFetcher.PIT_COVERAGE_START
    assert any("ZZZZ" in message for message in captured_messages)


def test_get_symbols_as_of_point_in_time_correctness(mock_universe_fetchers, tmp_path):
    config = _make_config(tmp_path)
    catalog = UniverseCatalog(config).build()

    before = catalog.get_symbols_as_of("sp500_constituent", "2020-12-20")
    on_date = catalog.get_symbols_as_of("sp500_constituent", "2020-12-21")

    assert "TSLA" not in before
    assert "TSLA" in on_date


def test_get_symbols_as_of_rejects_pre_1976_dates(mock_universe_fetchers, tmp_path):
    config = _make_config(tmp_path)
    catalog = UniverseCatalog(config).build()

    with pytest.raises(ValueError):
        catalog.get_symbols_as_of("sp500_constituent", "1970-01-01")


def test_get_symbols_as_of_nasdaq_all_uses_tiingo_dates(mock_universe_fetchers, tmp_path):
    config = _make_config(tmp_path)
    catalog = UniverseCatalog(config).build()

    symbols = catalog.get_symbols_as_of("nasdaq_all", "2021-01-01")

    assert "DELISTED1" not in symbols  # end_date 2020-01-01 is before as_of_date
    assert "AAPL" in symbols


def test_wikipedia_parse_failure_falls_back_to_cache(
    monkeypatch, mock_universe_fetchers, tmp_path
):
    fetcher = SP500MembershipFetcher(cache_dir=str(tmp_path / "_cache"))

    # Pre-populate the cache with a valid, larger snapshot than the broken
    # response we're about to simulate.
    good_snapshot = pl.DataFrame(
        {
            "effective_date": ["1990-01-01", "1991-01-01", "1992-01-01"],
            "added_ticker": ["A", "B", "C"],
            "removed_ticker": [None, None, None],
        }
    )
    fetcher._cache_path.parent.mkdir(parents=True, exist_ok=True)
    good_snapshot.write_parquet(fetcher._cache_path)
    cache_mtime_before = fetcher._cache_path.stat().st_mtime_ns
    cache_bytes_before = fetcher._cache_path.read_bytes()

    def broken_get(url, *args, **kwargs):
        class BrokenResponse:
            status_code = 200
            text = "<html><body><p>no table here</p></body></html>"
            content = b""

            def raise_for_status(self) -> None:
                pass

        if url == SP500MembershipFetcher.CHANGES_URL:
            return BrokenResponse()
        raise AssertionError(f"Unexpected URL requested in test: {url}")

    monkeypatch.setattr("acquisition.universe.requests.get", broken_get)

    result = fetcher.fetch_changes()

    assert result.shape[0] == good_snapshot.shape[0]
    assert fetcher._cache_path.stat().st_mtime_ns == cache_mtime_before
    assert fetcher._cache_path.read_bytes() == cache_bytes_before


def test_nasdaq100_build_intervals_reconstructs_membership(
    mock_universe_fetchers, tmp_path
):
    """End-to-end Nasdaq-100 interval reconstruction through the shared
    `IndexMembershipFetcher` base: mocked anchor HTML + mocked change-log HTML
    -> `[symbol, start_date, end_date]`, with the S&P 500 path unchanged.
    """
    fetcher = Nasdaq100MembershipFetcher(cache_dir=str(tmp_path))
    intervals = fetcher.build_intervals()

    assert intervals.columns == ["symbol", "start_date", "end_date"]

    logi = intervals.filter(pl.col("symbol") == "LOGI").to_dicts()
    assert len(logi) == 1
    assert logi[0]["start_date"] == "2007-02-01"
    assert logi[0]["end_date"] == "2018-03-02"

    newmem = intervals.filter(pl.col("symbol") == "NEWMEM").to_dicts()
    assert len(newmem) == 1
    assert newmem[0]["end_date"] is None

    assert intervals["start_date"].min() >= Nasdaq100MembershipFetcher.PIT_COVERAGE_START
