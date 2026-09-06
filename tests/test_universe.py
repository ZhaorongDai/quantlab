"""Tests for acquisition/universe.py -- point-in-time correctness,
left-censoring, re-entry, and graceful degradation.

No test in this module makes a real network call: all `requests.get` calls
are patched by the `mock_universe_fetchers` fixture in `tests/conftest.py`.
"""

import typing

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
from enums.data import UniverseCategory


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


# ---------------------------------------------------------------------------
# Nasdaq-100 locks (03.1-02-PLAN.md Task 2 / 03.1-RESEARCH.md Findings 2-5)
#
# Single-use HTML constructors stay module-local per 03.1-PATTERNS.md section 6
# ("shared data/mocks -> conftest.py; per-module constructors -> module-local").
# The shared 102-row anchor and 4-row change log live in `tests/conftest.py`.
# ---------------------------------------------------------------------------


def _ndx_anchor_html(symbols: list[str]) -> str:
    """Build a stockanalysis.com-shaped anchor page carrying exactly
    `symbols`. Used only to simulate a structurally-drifted (truncated) page.
    """
    body_rows = "\n".join(
        f"<tr><td>{i}</td><td>{sym}</td><td>{sym} Inc.</td>"
        f"<td>1.00B</td><td>10.00</td><td>0.10%</td><td>500.00M</td></tr>"
        for i, sym in enumerate(symbols, start=1)
    )
    return (
        "<html><body><table><thead><tr>"
        "<th>No.</th><th>Symbol</th><th>Company Name</th><th>Market Cap</th>"
        "<th>Stock Price</th><th>% Change</th><th>Revenue</th>"
        f"</tr></thead><tbody>{body_rows}</tbody></table></body></html>"
    )


def _ndx_changes_html(rows: list[tuple[str, str, str, str, str, str]]) -> str:
    """Build a Nasdaq-100 change-log page (six flat columns after read_html)
    carrying exactly `rows`. Used only to simulate a shrunken table.
    """
    body_rows = "\n".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return (
        "<html><body><table><thead>"
        '<tr><th rowspan="2">Date</th><th colspan="2">Added</th>'
        '<th colspan="2">Removed</th><th rowspan="2">Reason</th></tr>'
        "<tr><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th></tr>"
        f"</thead><tbody>{body_rows}</tbody></table></body></html>"
    )


class _FakeResponse:
    status_code = 200
    content = b""

    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self) -> None:
        pass


def test_nasdaq100_anchor_has_multiple_share_classes_not_exactly_one_hundred(
    mock_universe_fetchers, tmp_path
):
    """RESEARCH Finding 5: the live Nasdaq-100 anchor probed at **102** rows,
    not 100, because the index carries multiple share classes for some issuers
    (GOOGL/GOOG, FOX/FOXA). An `== 100` assertion is wrong against correct
    data, so this test asserts 102 and names both share-class pairs.

    Also locks the synthesised `date_added` column: neither real anchor source
    carries one, so `fetch_anchor()` must supply an explicit all-null column
    for `reconstruct_intervals()`'s `anchor_date_added.get(sym) or
    PIT_COVERAGE_START` fallback to fire instead of raising `KeyError`.
    """
    anchor = Nasdaq100MembershipFetcher(cache_dir=str(tmp_path)).fetch_anchor()

    assert anchor.height == 102
    symbols = set(anchor["symbol"].to_list())
    assert {"GOOGL", "GOOG"} <= symbols
    assert {"FOX", "FOXA"} <= symbols

    assert "date_added" in anchor.columns
    assert anchor["date_added"].null_count() == anchor.height


def test_nasdaq100_one_sided_change_rows_reconstruct_independently(
    mock_universe_fetchers, tmp_path
):
    """RESEARCH Finding 3: 18 of the live page's 226 rows are add-only and 16
    are drop-only. `reconstruct_intervals()` guards each side independently,
    so neither shape needs new logic -- this test proves that still holds
    after the `IndexMembershipFetcher` extraction.

    `NEWMEM` comes from the add-only row; `GONE1` from the drop-only row,
    which is ALSO left-censored (never added anywhere in the change log), so
    it must start at `PIT_COVERAGE_START` and emit a WARNING naming it.
    """
    fetcher = Nasdaq100MembershipFetcher(cache_dir=str(tmp_path))

    # loguru does not propagate to stdlib `logging` (pytest's `caplog`) --
    # attach a temporary in-memory sink instead (see the S&P 500 test above).
    captured_messages: list[str] = []
    sink_id = logger.add(captured_messages.append, level="WARNING", format="{message}")
    try:
        intervals = fetcher.build_intervals()
    finally:
        logger.remove(sink_id)

    newmem = intervals.filter(pl.col("symbol") == "NEWMEM").to_dicts()
    assert len(newmem) == 1
    assert newmem[0]["start_date"] == "2011-01-03"

    gone1 = intervals.filter(pl.col("symbol") == "GONE1").to_dicts()
    assert len(gone1) == 1
    assert gone1[0]["start_date"] == Nasdaq100MembershipFetcher.PIT_COVERAGE_START
    assert gone1[0]["end_date"] == "2015-06-15"
    assert any("GONE1" in message for message in captured_messages)


def test_nasdaq100_anchor_rejects_a_structurally_drifted_page(monkeypatch, tmp_path):
    """T-03.1-02-02. The Nasdaq-100 anchor is the least stable input in this
    phase: unlike the S&P 500's GitHub-hosted CSV it is scraped from a
    commercial page (stockanalysis.com) whose markup can change without notice,
    and it has no cache-fallback path of its own.

    A silently truncated anchor would close every membership not mentioned in
    the change log and quietly reintroduce exactly the survivorship bias this
    data layer exists to remove -- so a drifted page must raise, not degrade.
    """
    fetcher = Nasdaq100MembershipFetcher(cache_dir=str(tmp_path))
    drifted = _ndx_anchor_html(["AAPL", "MSFT", "NVDA"])

    def fake_get(url, *args, **kwargs):
        if url == Nasdaq100MembershipFetcher.ANCHOR_URL:
            return _FakeResponse(drifted)
        raise AssertionError(f"Unexpected URL requested in test: {url}")

    monkeypatch.setattr("acquisition.universe.requests.get", fake_get)

    with pytest.raises(ValueError):
        fetcher.fetch_anchor()


def test_nasdaq100_changes_parse_failure_falls_back_to_cache_without_overwriting_it(
    monkeypatch, mock_universe_fetchers, tmp_path
):
    """T-03.1-02-01, Nasdaq-100 counterpart of
    `test_wikipedia_parse_failure_falls_back_to_cache`. Proves the base class's
    non-destructive fallback survived the extraction for BOTH indices: on a
    parse failure the cached snapshot is returned and the cache file is left
    byte-for-byte untouched, so one bad parse cannot poison future runs.
    """
    fetcher = Nasdaq100MembershipFetcher(cache_dir=str(tmp_path / "_cache"))

    # Build a real cache through the normal (mocked-but-valid) path first.
    good = fetcher.fetch_changes()
    assert fetcher._cache_path.exists()
    cache_mtime_before = fetcher._cache_path.stat().st_mtime_ns
    cache_bytes_before = fetcher._cache_path.read_bytes()

    def broken_get(url, *args, **kwargs):
        if url == Nasdaq100MembershipFetcher.CHANGES_URL:
            return _FakeResponse("<html><body><p>no table here</p></body></html>")
        raise AssertionError(f"Unexpected URL requested in test: {url}")

    monkeypatch.setattr("acquisition.universe.requests.get", broken_get)

    result = fetcher.fetch_changes()

    assert result.shape[0] == good.shape[0]
    assert fetcher._cache_path.stat().st_mtime_ns == cache_mtime_before
    assert fetcher._cache_path.read_bytes() == cache_bytes_before


def test_nasdaq100_row_count_monotonicity_guard_rejects_a_shrunken_table(
    monkeypatch, mock_universe_fetchers, tmp_path
):
    """T-03.1-02-01, this phase's main parse-integrity defence. Index
    memberships only ever close -- they never retroactively vanish -- so a
    live change table with fewer rows than the cached snapshot is a parse
    failure or schema drift, never new truth. The fetcher must fall back to
    the cache and leave it byte-for-byte unchanged rather than accept the
    shrunken table.
    """
    fetcher = Nasdaq100MembershipFetcher(cache_dir=str(tmp_path / "_cache"))

    good = fetcher.fetch_changes()
    assert good.shape[0] == 4
    cache_mtime_before = fetcher._cache_path.stat().st_mtime_ns
    cache_bytes_before = fetcher._cache_path.read_bytes()

    shrunken = _ndx_changes_html(
        [
            ("March 2, 2018", "TEMP1", "Temp One Inc.", "LOGI", "Logitech", "Reason"),
            ("February 1, 2007", "LOGI", "Logitech", "CMVT", "Comverse", "Reason"),
        ]
    )

    def shrunken_get(url, *args, **kwargs):
        if url == Nasdaq100MembershipFetcher.CHANGES_URL:
            return _FakeResponse(shrunken)
        raise AssertionError(f"Unexpected URL requested in test: {url}")

    monkeypatch.setattr("acquisition.universe.requests.get", shrunken_get)

    result = fetcher.fetch_changes()

    assert result.shape[0] == good.shape[0]
    assert fetcher._cache_path.stat().st_mtime_ns == cache_mtime_before
    assert fetcher._cache_path.read_bytes() == cache_bytes_before


def test_universe_category_literal_has_exactly_three_values():
    """03.1-CONTEXT.md D-02: `nasdaq100_constituent` is a THIRD category
    alongside `nasdaq_all`, never a replacement for it. Pinning the literal
    means neither value can drift away without a test noticing.
    """
    assert set(typing.get_args(UniverseCategory)) == {
        "nasdaq_all",
        "sp500_constituent",
        "nasdaq100_constituent",
    }


# ---------------------------------------------------------------------------
# Registry-driven catalog (03.1-04-PLAN.md Task 2, DATA-05/DATA-06)
# ---------------------------------------------------------------------------


def test_catalog_build_emits_all_three_categories(mock_universe_fetchers, tmp_path):
    """`build()` loops `UniverseCatalog.MEMBERSHIP_FETCHERS` and derives each
    category token from `cls.CATEGORY`, so registering an index is the whole
    cost of adding it to the reference table.

    Pinning the distinct set against `UniverseCategory` means a fetcher
    registered without its enum token -- or an enum token with no fetcher --
    fails here rather than producing a table that silently omits a category.
    """
    catalog = UniverseCatalog(_make_config(tmp_path)).build()

    categories = (
        catalog._backend.get_lazyframe()
        .select("category")
        .unique()
        .collect()["category"]
        .to_list()
    )

    assert set(categories) == set(typing.get_args(UniverseCategory))


def test_get_symbols_as_of_rejects_pre_2007_nasdaq100_dates(
    mock_universe_fetchers, tmp_path
):
    """The explicit-error half of DATA-05, and the Nasdaq-100 counterpart of
    `test_get_symbols_as_of_rejects_pre_1976_dates`.

    Before the registry the coverage guard hardcoded `sp500_constituent`, so
    this exact query would have returned an INCOMPLETE ROSTER with no error --
    and an incomplete roster is indistinguishable from a correct one to the
    backtest consuming it (T-03.1-04-01). Driving the guard from each
    fetcher's own `PIT_COVERAGE_START` means a category cannot exist without
    a boundary.
    """
    catalog = UniverseCatalog(_make_config(tmp_path)).build()

    with pytest.raises(ValueError):
        catalog.get_symbols_as_of("nasdaq100_constituent", "2005-01-01")


def test_get_symbols_as_of_nasdaq100_point_in_time_correctness(
    mock_universe_fetchers, tmp_path
):
    """Point-in-time correctness for the second index, mirroring the S&P 500
    test above: `NEWMEM` is added by the fixture's add-only 2011-01-03 row, so
    it must be absent the day before and present on the day itself.

    The closed-interval convention is what makes the effective date itself a
    membership day -- the same convention the daily panel densifies to.
    """
    catalog = UniverseCatalog(_make_config(tmp_path)).build()

    before = catalog.get_symbols_as_of("nasdaq100_constituent", "2011-01-02")
    on_date = catalog.get_symbols_as_of("nasdaq100_constituent", "2011-01-03")

    assert "NEWMEM" not in before
    assert "NEWMEM" in on_date


def test_nasdaq_all_has_no_coverage_boundary(mock_universe_fetchers, tmp_path):
    """03.1-CONTEXT.md D-02: `nasdaq_all` semantics are unchanged.

    `NasdaqUniverseFetcher` is deliberately NOT in `MEMBERSHIP_FETCHERS`: it
    is a full-exchange roster sourced from Tiingo's `supported_tickers.csv`,
    with per-symbol listing dates but no index-membership concept and so no
    point-in-time coverage start. The registry must therefore leave it
    boundary-free -- a 1970 query answers from the roster's own dates rather
    than raising.
    """
    catalog = UniverseCatalog(_make_config(tmp_path)).build()

    assert catalog.get_symbols_as_of("nasdaq_all", "1970-01-01") == []
