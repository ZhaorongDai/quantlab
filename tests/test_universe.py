"""Tests for acquisition/universe.py -- point-in-time correctness,
left-censoring, re-entry, and graceful degradation.

No test in this module makes a real network call: all `requests.get` calls
are patched by the `mock_universe_fetchers` fixture in `tests/conftest.py`.
"""

import io
import typing
import zipfile
from pathlib import Path

import polars as pl
import pytest
from loguru import logger

from acquisition.universe import (
    Nasdaq100MembershipFetcher,
    NasdaqUniverseFetcher,
    SP500MembershipFetcher,
    TiingoRosterFetcher,
    UniverseCatalog,
    USEquityUniverseFetcher,
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


def test_nasdaq_roster_guard_rejects_a_drifted_filter(monkeypatch, tmp_path):
    """CR-05. `fetch()` filters on exact-match string literals, so a casing or
    spelling change in Tiingo's feed yields ZERO rows without raising --
    `build()` then concatenates the empty frame happily and `save()` overwrites
    the previous good 10,000+-symbol table.

    Exercised at the guard's REAL value (`mock_universe_fetchers` lowers it, so
    this test deliberately does not use that fixture): a roster that survives
    the filter with fewer than MIN_ROSTER_ROWS rows must raise.
    """
    drifted_csv = (
        "ticker,exchange,assetType,priceCurrency,startDate,endDate\n"
        # "Nasdaq" rather than "NASDAQ" -- one casing change, whole roster gone.
        "AAPL,Nasdaq,Stock,USD,1980-12-12,\n"
        "MSFT,Nasdaq,Stock,USD,1986-03-13,\n"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("supported_tickers.csv", drifted_csv)
    payload = buffer.getvalue()

    class _ZipResponse:
        status_code = 200
        content = payload

        def raise_for_status(self) -> None:
            pass

    monkeypatch.setattr(
        "acquisition.universe.requests.get", lambda url, *a, **k: _ZipResponse()
    )

    with pytest.raises(ValueError, match="token vocabulary has drifted"):
        NasdaqUniverseFetcher().fetch()


def test_save_refuses_to_overwrite_with_a_degenerate_table(
    mock_universe_fetchers, tmp_path
):
    """CR-05, second half. `save()` overwrites `universe.parquet` in place, so
    a category that silently came back empty would destroy the previous good
    roster with no error anywhere in the pipeline -- the only symptom being
    downstream ingestion quietly resolving an empty symbol list.
    """
    config = _make_config(tmp_path)
    catalog = UniverseCatalog(config).build()

    # Drop one category, simulating a fetcher that silently returned nothing.
    surviving = (
        catalog._backend.get_lazyframe()
        .filter(pl.col("category") != "nasdaq100_constituent")
    )
    catalog._backend.to_internal(surviving)

    with pytest.raises(ValueError, match="have no rows"):
        catalog.save()

    assert not Path(config.output_path).exists()


def test_reconstruct_intervals_reentry(mock_universe_fetchers, tmp_path):
    fetcher = SP500MembershipFetcher(cache_dir=str(tmp_path))
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


def test_orphaned_open_interval_is_flagged_as_an_inferred_end(
    mock_universe_fetchers, tmp_path
):
    """WR-08. When a symbol has an open interval but is absent from the anchor,
    its end_date is set to `last_eff` -- the effective date of the last row
    ANYWHERE in the change log, which is not an observation about that symbol.
    The end is fabricated, and it was written into the persisted table exactly
    like an observed one, with only a logger.warning to distinguish it.

    `REENTRY` is re-added in 2000 by the fixture and never appears in the
    anchor CSV, so it takes that path.
    """
    fetcher = SP500MembershipFetcher(cache_dir=str(tmp_path))
    intervals = fetcher.reconstruct_intervals(
        fetcher.fetch_anchor(), fetcher.fetch_changes()
    )

    orphaned = (
        intervals.filter(pl.col("symbol") == "REENTRY")
        .sort("start_date")
        .to_dicts()[-1]
    )
    assert orphaned["end_date"] is not None
    assert orphaned["end_date_is_inferred"] is True

    # An end that really was observed is not flagged.
    observed = intervals.filter(pl.col("symbol") == "AIV").to_dicts()[0]
    assert observed["end_date"] == "2020-12-21"
    assert observed["end_date_is_inferred"] is False


def test_inferred_end_flag_survives_the_parquet_round_trip(
    mock_universe_fetchers, tmp_path
):
    """WR-08. The flag is only useful if it reaches the PERSISTED table -- that
    is the artefact a downstream consumer reads.
    """
    config = _make_config(tmp_path)
    UniverseCatalog(config).build().save()

    reloaded = (
        UniverseCatalog.load(config)
        ._backend.get_lazyframe()
        .filter(pl.col("end_date_is_inferred"))
        .collect()
    )

    assert "REENTRY" in reloaded["symbol"].to_list()


def test_reconstruct_intervals_left_censored(mock_universe_fetchers, tmp_path):
    fetcher = SP500MembershipFetcher(cache_dir=str(tmp_path))
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


def test_reconstruct_intervals_reopens_a_current_member_whose_last_event_was_a_removal(
    tmp_path,
):
    """CR-01. The change log and the anchor disagree in three directions, not
    two. This is the third: the log's LAST event for a symbol is a removal,
    but the anchor still lists it as a current constituent -- i.e. the log is
    missing a re-addition. The live Nasdaq-100 log is known to be asymmetric
    (16 drop-only rows), so this is a real shape.

    Before the fix the symbol was in `seen_symbols` (from the closed removal)
    and so was skipped by the anchor-only loop, leaving it persisted as a
    FORMER member with no warning -- the mirror-image inconsistency warns. The
    consequence is a `get_symbols_as_of(category, today)` that omits a current
    constituent, undetectable downstream.
    """
    fetcher = SP500MembershipFetcher(cache_dir=str(tmp_path))
    anchor = pl.DataFrame(
        {"symbol": ["CURR"], "date_added": [None]},
        schema={"symbol": pl.String, "date_added": pl.String},
    )
    changes = pl.DataFrame(
        {
            "effective_date": ["1990-01-01", "2005-01-01"],
            "added_ticker": ["CURR", None],
            "removed_ticker": [None, "CURR"],
        }
    )

    captured_messages: list[str] = []
    sink_id = logger.add(captured_messages.append, level="WARNING", format="{message}")
    try:
        intervals = fetcher.reconstruct_intervals(anchor, changes)
    finally:
        logger.remove(sink_id)

    rows = intervals.filter(pl.col("symbol") == "CURR").sort("start_date").to_dicts()
    assert [row["end_date"] for row in rows][-1] is None, (
        "the anchor is authoritative for 'is a member today'; CURR must end "
        f"with an OPEN interval, got {rows}"
    )
    assert any("CURR" in message for message in captured_messages), (
        "re-opening a membership the change log never re-added must not be silent"
    )


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


def test_get_symbols_as_of_rejects_an_unknown_category(mock_universe_fetchers, tmp_path):
    """CR-04. An unknown category returned `[]` -- a LEGITIMATE value that
    `test_nasdaq_all_has_no_coverage_boundary` asserts for a real query -- so
    a typo was indistinguishable from "no members" and silently ingested
    nothing.
    """
    catalog = UniverseCatalog(_make_config(tmp_path)).build()

    with pytest.raises(ValueError, match="Unknown universe category"):
        catalog.get_symbols_as_of("sp500", "2020-01-01")  # correct token is sp500_constituent


def test_get_symbols_as_of_rejects_a_non_iso_date(mock_universe_fetchers, tmp_path):
    """CR-04. `as_of_date` reaches this function straight off
    `ingest_tiingo.py`'s `--as-of-date` CLI argument. Dates are compared
    LEXICOGRAPHICALLY against ISO strings, so `"01/01/2024"` does not merely
    fail to match -- `"1980-12-12" <= "01/01/2024"` is False -- and a mistyped
    date silently ingested nothing instead of the requested index.
    """
    catalog = UniverseCatalog(_make_config(tmp_path)).build()

    with pytest.raises(ValueError, match="ISO YYYY-MM-DD"):
        catalog.get_symbols_as_of("nasdaq_all", "01/01/2024")


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

    assert intervals.columns == [
        "symbol",
        "start_date",
        "end_date",
        # WR-08: marks an interval end this reconstruction FABRICATED rather
        # than observed, so a consumer can tell the two apart.
        "end_date_is_inferred",
    ]

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


def test_na_tickered_change_rows_survive_parsing(tmp_path):
    """CR-02. `NA` is a real, historically-listed US equity ticker and is also
    in pandas' default NA vocabulary. Under `read_html`'s default coercion it
    became NaN -- the exact sentinel `reconstruct_intervals()` reads as "no
    change on this side" -- so the add/remove event was silently DISCARDED.

    Both directions are asserted: the `NA` ticker survives as itself, and a
    genuinely blank cell still becomes a real `None`.
    """
    fetcher = Nasdaq100MembershipFetcher(cache_dir=str(tmp_path))
    html = _ndx_changes_html(
        [
            ("February 1, 2007", "NA", "Nabors Industries", "CMVT", "Comverse", "R"),
            ("March 2, 2018", "", "", "NA", "Nabors Industries", "R"),
        ]
    )

    # Asserted after `pl.from_pandas`, which is the shape
    # `reconstruct_intervals()` actually consumes and where "no change on this
    # side" must be a genuine null.
    parsed = pl.from_pandas(fetcher._parse_changes_table(html))

    assert parsed["added_ticker"].to_list() == ["NA", None]
    assert parsed["removed_ticker"].to_list() == ["CMVT", "NA"]


def test_na_tickered_anchor_row_is_not_turned_into_a_nan_symbol(monkeypatch, tmp_path):
    """CR-02, anchor half. `str(sym)` on a coerced NaN produced the literal
    string `"nan"`, which entered the anchor as a fabricated permanent
    constituent -- an always-True column in the densified panel -- while the
    real ticker `NA` disappeared.
    """
    fetcher = Nasdaq100MembershipFetcher(cache_dir=str(tmp_path))
    page = _ndx_anchor_html(["NA"] + [f"NDX{i:03d}" for i in range(1, 60)])

    monkeypatch.setattr(
        "acquisition.universe.requests.get",
        lambda url, *a, **k: _FakeResponse(page),
    )

    symbols = set(fetcher.fetch_anchor()["symbol"].to_list())

    assert "NA" in symbols
    assert "nan" not in symbols


def test_unparseable_effective_date_names_the_offending_rows(tmp_path):
    """WR-11. `pd.to_datetime` was called with no `format=` and no `errors=`,
    so a single unparseable cell -- Wikipedia routinely carries footnote
    markers and date ranges in date columns -- raised DateParseError, which
    `fetch_changes()`'s broad `except Exception` converted into a permanent,
    near-silent fallback to the stale cache.

    The bad cell must now be named in the error rather than silently dropped
    or laundered into a cache fallback.
    """
    fetcher = Nasdaq100MembershipFetcher(cache_dir=str(tmp_path))
    html = _ndx_changes_html(
        [
            ("February 1, 2007", "LOGI", "Logitech", "CMVT", "Comverse", "R"),
            ("not a date[1]", "TEMP1", "Temp One", "", "", "R"),
        ]
    )

    with pytest.raises(ValueError, match="unparseable effective_date"):
        fetcher._parse_changes_table(html)


def _ndx_changes_html_with_swapped_groups(
    rows: list[tuple[str, str, str, str, str, str]],
) -> str:
    """A Nasdaq-100 change-log page whose `Added`/`Removed` header GROUPS are
    swapped -- same six columns, same column count, only the order differs.
    """
    body_rows = "\n".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return (
        "<html><body><table><thead>"
        '<tr><th rowspan="2">Date</th><th colspan="2">Removed</th>'
        '<th colspan="2">Added</th><th rowspan="2">Reason</th></tr>'
        "<tr><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th></tr>"
        f"</thead><tbody>{body_rows}</tbody></table></body></html>"
    )


def test_changes_parse_rejects_a_reordered_source_header(monkeypatch, tmp_path):
    """CR-03. A header REORDER has the same column count as a correct header,
    so the old implicit `Length mismatch` guard never fired on it -- and the
    base's `required_columns` check could not either, because each subclass
    ASSIGNED those very names positionally before returning. The result was
    that a swapped `Added`/`Removed` group was accepted and every event
    recorded exactly inverted, with no error and no warning.

    The table is now SELECTED by matching its flattened source header, and the
    ticker columns are read by name off that verified header, so a reorder
    must raise instead of silently inverting add/remove.
    """
    fetcher = Nasdaq100MembershipFetcher(cache_dir=str(tmp_path))
    swapped = _ndx_changes_html_with_swapped_groups(
        [("February 1, 2007", "CMVT", "Comverse", "LOGI", "Logitech", "Reason")]
    )

    monkeypatch.setattr(
        "acquisition.universe.requests.get",
        lambda url, *a, **k: _FakeResponse(swapped),
    )

    with pytest.raises(ValueError, match="expected change-log header"):
        fetcher._parse_changes_table(swapped)


def test_changes_table_is_selected_by_header_not_by_position(monkeypatch, tmp_path):
    """CR-03, aggravating factor. `Nasdaq100MembershipFetcher` used to take
    `tables[0]` with no selector at all, so any table Wikipedia inserted ahead
    of the change log became the change log. Selecting by header identity
    makes the decoy inert.
    """
    fetcher = Nasdaq100MembershipFetcher(cache_dir=str(tmp_path))
    decoy = (
        "<html><body>"
        "<table><thead><tr><th>Rank</th><th>Company</th><th>Weight</th></tr>"
        "</thead><tbody><tr><td>1</td><td>Apple</td><td>8%</td></tr></tbody>"
        "</table>"
    )
    real = _ndx_changes_html(
        [("February 1, 2007", "LOGI", "Logitech", "CMVT", "Comverse", "Reason")]
    )
    combined = decoy + real[len("<html><body>") :]

    parsed = fetcher._parse_changes_table(combined)

    assert parsed["added_ticker"].tolist() == ["LOGI"]
    assert parsed["removed_ticker"].tolist() == ["CMVT"]


def test_every_universe_category_is_reachable_from_the_cli():
    """WR-05. The phase added a third UniverseCategory and registered its
    fetcher, but ingest_tiingo.py's _UNIVERSE_CATEGORY_MAP and its
    `--universe` choices were a SECOND hardcoded list that was not extended --
    so nasdaq100_constituent was produced into universe.parquet and could
    never be selected from the only CLI that consumes the table.

    Pinning the map against the enum means a fourth category cannot be added
    without becoming reachable, and the `choices` are derived from the map so
    the two can no longer disagree.
    """
    import ingest_tiingo

    assert set(ingest_tiingo._UNIVERSE_CATEGORY_MAP.values()) == set(
        typing.get_args(UniverseCategory)
    )
    choices = ingest_tiingo._build_arg_parser()._option_string_actions[
        "--universe"
    ].choices
    assert set(choices) == set(ingest_tiingo._UNIVERSE_CATEGORY_MAP)


def test_universe_category_literal_has_exactly_four_values():
    """03.1-CONTEXT.md D-02: `nasdaq100_constituent` is a category alongside
    `nasdaq_all`, never a replacement for it. 260906-0iy D-01/D-02 adds
    `us_all` on the same footing: it is a SUPERSET of `nasdaq_all` and both
    are deliberately retained. Pinning the literal means no value can drift
    away -- or be quietly redefined as an alias of another -- unnoticed.
    """
    assert set(typing.get_args(UniverseCategory)) == {
        "nasdaq_all",
        "us_all",
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


def test_catalog_round_trips_through_parquet(mock_universe_fetchers, tmp_path):
    """WR-10. Every other catalog test calls .build() and then reads
    catalog._backend directly, so save() -> load() -> get_symbols_as_of() --
    exactly the sequence refresh_us_equity_universe.py writes and
    ingest_tiingo.py reads -- was entirely untested.

    start_date/end_date are compared as STRINGS, so a dtype change across the
    parquet round trip would silently break point-in-time correctness in
    production while every in-memory test stayed green.
    """
    config = _make_config(tmp_path)
    UniverseCatalog(config).build().save()

    reloaded = UniverseCatalog.load(config)

    assert "TSLA" in reloaded.get_symbols_as_of("sp500_constituent", "2020-12-21")
    assert "TSLA" not in reloaded.get_symbols_as_of("sp500_constituent", "2020-12-20")
    assert "NEWMEM" in reloaded.get_symbols_as_of("nasdaq100_constituent", "2011-01-03")
    assert "AAPL" in reloaded.get_symbols_as_of("nasdaq_all", "2021-01-01")


def test_build_refuses_stale_snapshots_unless_explicitly_allowed(
    monkeypatch, mock_universe_fetchers, tmp_path
):
    """WR-01. The cached-snapshot fallback is deliberate, but build()/save()
    persisted it into universe.parquet indistinguishably from a fresh
    reconstruction -- so a permanently-broken source froze the universe at the
    cache date with only a logger.error to record it.
    """
    config = _make_config(tmp_path)
    # Seed both caches through the working mocks.
    UniverseCatalog(config).build()

    real_get = mock_universe_fetchers

    def broken_changes(url, *args, **kwargs):
        if url == SP500MembershipFetcher.CHANGES_URL:
            return _FakeResponse("<html><body><p>no table here</p></body></html>")
        return real_get(url, *args, **kwargs)

    monkeypatch.setattr("acquisition.universe.requests.get", broken_changes)

    with pytest.raises(ValueError, match="stale cached snapshots"):
        UniverseCatalog(config).build()

    # The degradation stays available, but only as an explicit decision.
    catalog = UniverseCatalog(config).build(allow_stale=True)
    assert "TSLA" in catalog.get_symbols_as_of("sp500_constituent", "2020-12-21")


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


# ---------------------------------------------------------------------------
# Full-US-market roster: `us_all` (260906-0iy Task 1, D-01/D-02/D-05)
# ---------------------------------------------------------------------------


def test_us_equity_fetcher_filters_nyse_nasdaq_amex_and_excludes_others(
    mock_universe_fetchers,
):
    """D-01: NYSE + NASDAQ + AMEX common stock priced in USD.

    The AMEX arrives under TWO tokens (`AMEX` and `NYSE MKT`) because Tiingo
    never re-labelled its historical rows across the exchange's rename
    history; omitting either silently drops ~224 real tickers. `NYSE ARCA`
    shares a prefix but is a DIFFERENT exchange (predominantly ETFs) and is
    deliberately out of scope.
    """
    result = USEquityUniverseFetcher().fetch()

    symbols = set(result["symbol"].to_list())

    assert {"AAPL", "MSFT", "DELISTED1"} <= symbols  # NASDAQ
    assert {"NYSE1", "NYSE2"} <= symbols  # NYSE
    assert "AMEX1" in symbols  # AMEX token
    assert "MKT1" in symbols  # NYSE MKT token -- same exchange, renamed
    assert "DUAL1" in symbols  # dual-listed
    assert "OLD1" in symbols  # pre-2006 delisting is a ROSTER member...

    assert "ARCA1" not in symbols  # different exchange
    assert "ETF1" not in symbols  # wrong assetType
    assert "EURO1" not in symbols  # wrong priceCurrency


def test_nasdaq_roster_exchange_filter_and_symbol_set_are_unchanged(
    mock_universe_fetchers,
):
    """D-02, machine-checked by direct equality rather than by grep.

    `USEquityUniverseFetcher` is a NEW SIBLING of the NASDAQ-only roster, not
    a widening of it. Extracting the shared body onto `TiingoRosterFetcher`
    must be behaviour-preserving: the constant is byte-for-byte what it was,
    and `fetch()` still returns exactly the NASDAQ subset.
    """
    assert NasdaqUniverseFetcher.EXCHANGE_FILTER == ("NASDAQ",)
    assert NasdaqUniverseFetcher.CATEGORY == "nasdaq_all"
    assert issubclass(NasdaqUniverseFetcher, TiingoRosterFetcher)

    symbols = set(NasdaqUniverseFetcher().fetch()["symbol"].to_list())
    assert symbols == {"AAPL", "MSFT", "DELISTED1"}


def test_us_equity_roster_guard_rejects_a_drifted_filter(monkeypatch, tmp_path):
    """The same safety envelope as `NasdaqUniverseFetcher.MIN_ROSTER_ROWS`,
    sized to its own magnitude (8000 vs the observed ~16,138 rows).

    Exercised at the guard's REAL value -- `mock_universe_fetchers` lowers it,
    so this test deliberately does not use that fixture. A token-vocabulary
    drift that silently zeroes the filter must raise rather than let `save()`
    overwrite a good 15,000-symbol reference table with a truncated roster.
    """
    drifted_csv = (
        "ticker,exchange,assetType,priceCurrency,startDate,endDate\n"
        # "Nyse"/"Nasdaq" rather than the real tokens -- one casing change.
        "AAPL,Nasdaq,Stock,USD,1980-12-12,\n"
        "GE,Nyse,Stock,USD,1962-01-02,\n"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("supported_tickers.csv", drifted_csv)
    payload = buffer.getvalue()

    class _ZipResponse:
        status_code = 200
        content = payload

        def raise_for_status(self) -> None:
            pass

    monkeypatch.setattr(
        "acquisition.universe.requests.get", lambda url, *a, **k: _ZipResponse()
    )

    assert USEquityUniverseFetcher.MIN_ROSTER_ROWS == 8000
    with pytest.raises(ValueError, match="token vocabulary has drifted"):
        USEquityUniverseFetcher().fetch()


def test_roster_fetchers_registry_is_disjoint_from_membership_fetchers():
    """Roster fetchers live in their OWN registry precisely because they have
    no membership-interval semantics and no coverage start.

    Registering either in `MEMBERSHIP_FETCHERS` would impose a
    `PIT_COVERAGE_START` boundary neither roster may have (D-02).
    """
    assert set(UniverseCatalog.ROSTER_FETCHERS) == {
        NasdaqUniverseFetcher,
        USEquityUniverseFetcher,
    }
    assert not set(UniverseCatalog.ROSTER_FETCHERS) & set(
        UniverseCatalog.MEMBERSHIP_FETCHERS
    )


def test_catalog_build_emits_all_four_categories(mock_universe_fetchers, tmp_path):
    """Both registries are looped, so every `UniverseCategory` token has a
    fetcher and every fetcher has a token -- a fetcher registered without its
    enum token (or vice versa) fails here rather than producing a table that
    silently omits a category.
    """
    catalog = UniverseCatalog(_make_config(tmp_path)).build()

    categories = set(
        catalog._backend.get_lazyframe()
        .select("category")
        .unique()
        .collect()["category"]
        .to_list()
    )

    assert categories == set(typing.get_args(UniverseCategory))
    assert catalog.known_categories() == set(typing.get_args(UniverseCategory))
    assert "us_all" in categories


def test_get_symbols_in_range_keeps_post_2006_delistings(
    mock_universe_fetchers, tmp_path
):
    """D-05, the whole point of the layer: interval OVERLAP, not
    point-in-time membership.

    A backfill over 2006-01-01..today must keep every ticker that traded at
    ANY point in the window -- including the ~6.9k that delisted inside it.
    Dropping them reintroduces exactly the survivorship bias this exists to
    avoid. Only a ticker whose `endDate` precedes the window start may go.
    """
    catalog = UniverseCatalog(_make_config(tmp_path)).build()

    symbols = catalog.get_symbols_in_range("us_all", "2006-01-01", "2026-09-06")

    assert "DELISTED1" in symbols  # ended 2020-01-01, INSIDE the window
    assert "AAPL" in symbols  # still listed
    assert "OLD1" not in symbols  # ended 1997-06-30, before the window


def test_get_symbols_in_range_deduplicates_a_dual_listed_ticker(
    mock_universe_fetchers, tmp_path
):
    """~700 real tickers carry more than one exchange row (re-use / venue
    migration), so the query must `.unique()` on symbol exactly as
    `get_symbols_as_of` already does -- otherwise the roster handed to the
    acquisition layer would fetch those symbols twice.
    """
    catalog = UniverseCatalog(_make_config(tmp_path)).build()

    symbols = catalog.get_symbols_in_range("us_all", "2006-01-01", "2026-09-06")

    assert symbols.count("DUAL1") == 1
    assert len(symbols) == len(set(symbols))


def test_get_symbols_in_range_rejects_an_unknown_category(
    mock_universe_fetchers, tmp_path
):
    """Same reasoning `get_symbols_as_of` already applies: `[]` is a
    LEGITIMATE return value, so a typo'd category must raise rather than
    silently ingest nothing.
    """
    catalog = UniverseCatalog(_make_config(tmp_path)).build()

    with pytest.raises(ValueError, match="Unknown universe category"):
        catalog.get_symbols_in_range("us_alll", "2006-01-01", "2026-09-06")


def test_get_symbols_in_range_rejects_a_non_iso_date(
    mock_universe_fetchers, tmp_path
):
    """Dates are compared LEXICOGRAPHICALLY against ISO strings, so a non-ISO
    value does not merely fail to match -- it compares wrong and returns a
    plausible, silently incorrect roster. Both ends are validated.
    """
    catalog = UniverseCatalog(_make_config(tmp_path)).build()

    with pytest.raises(ValueError, match="ISO YYYY-MM-DD"):
        catalog.get_symbols_in_range("us_all", "01/01/2006", "2026-09-06")
    with pytest.raises(ValueError, match="ISO YYYY-MM-DD"):
        catalog.get_symbols_in_range("us_all", "2006-01-01", "09/06/2026")


def test_us_all_has_no_coverage_boundary(mock_universe_fetchers, tmp_path):
    """`us_all` is boundary-free for the same reason `nasdaq_all` is (D-02):
    it is a full-market ROSTER with per-symbol listing dates, not index
    membership, so it has no left-censored change log and no coverage start.
    A pre-listing query answers from the roster's own dates rather than
    raising.
    """
    catalog = UniverseCatalog(_make_config(tmp_path)).build()

    assert catalog.get_symbols_as_of("us_all", "1970-01-01") == []
    assert "AMEX1" in catalog.get_symbols_as_of("us_all", "2000-01-01")
