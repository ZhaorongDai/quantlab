"""StockFit's refresh over a full history: the backfill across days, the daily dirty set, the weekly re-pull.

Seam 1 of the StockFit spec, the raw-tier refresh entry, with HTTP replaced
by ``tests.stockfit_fixtures.FakeVendor``. The planner that orders the
endpoint x entity jobs is tested only through the refresh; what lands on disk
is read back through ``quantlab.dataset.stockfit.raw``.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from tests.stockfit_fixtures import (
    GOOGL,
    KEY,
    PLUG,
    STATEMENT_PATHS,
    TSM,
    TWTR,
    FakeClock,
    FakeVendor,
    all_but_13f,
    company_row,
    filing_row,
    fixture,
    list_requests,
    statement_endpoints,
)


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("STOCKFIT_API_KEY", KEY)


@pytest.fixture
def clock():
    return FakeClock()


def _refresh(tmp_path, vendor, clock, **kwargs):
    from quantlab.acquisition.stockfit.refresh import refresh_raw_tier

    client = vendor.client()
    kwargs.setdefault("endpoints", all_but_13f())
    return refresh_raw_tier(tmp_path, client=client, **kwargs)


def _tier(root) -> dict:
    """Every entity of the raw tier: its newest pull and how many pulls it has."""
    from quantlab.dataset.stockfit import raw

    return {
        (path.name, entity): (raw.read_pull(root, path.name, entity), len(raw.pulls(root, path.name, entity)))
        for path in sorted(root.iterdir()) if path.is_dir()
        for entity in raw.entities(root, path.name)
    }


def _statement_calls(vendor) -> list[dict]:
    return [params for path in STATEMENT_PATHS for params in vendor.requested(path)]


def test_a_capped_backfill_spans_several_runs_and_ends_with_the_raw_tier_of_one_uncapped_run(env, tmp_path, clock):
    uncapped = tmp_path / "uncapped"
    own = FakeClock()  # the vendor and the client share a clock, so the minute window rolls
    assert not _refresh(uncapped, FakeVendor(own), own).failed

    capped = tmp_path / "capped"
    # every market-wide list once a day (company and fund lists, feeds, forward lists), and 28 for the rest
    vendor = FakeVendor(clock, per_day=list_requests(all_but_13f()) + 28)
    runs = []
    while True:
        result = _refresh(capped, vendor, clock)
        runs.append(result)
        assert not result.failed, result.summary()
        assert 429 not in vendor.statuses
        if "deferred" not in result.summary():
            break
        assert len(runs) < 20
        clock.advance(days=1)

    assert len(runs) > 2
    assert _tier(capped / "stockfit") == _tier(uncapped / "stockfit")


def test_a_429_makes_the_run_wait_and_retry_and_never_fail(env, tmp_path, clock):
    from quantlab.dataset.stockfit import raw

    throttled = (429, b'{"error":"Too Many Requests"}', {"X-RateLimit-Remaining-Minute": "0", "X-RateLimit-Remaining-Day": "900"})
    vendor = FakeVendor(clock, replies=[throttled] * 8)  # more 429s in a row than the client's retries
    result = _refresh(tmp_path, vendor, clock, endpoints=["income_statement_quarter"], ciks=[PLUG])

    assert not result.failed, result.summary()
    assert clock.sleeps == [60.0] * 8
    assert raw.read_pull(tmp_path / "stockfit", "income_statement_quarter", str(PLUG))


def test_a_day_with_one_new_10k_re_pulls_only_that_companys_filing_driven_endpoints(env, tmp_path, clock):
    from quantlab.dataset.stockfit import raw

    vendor = FakeVendor(clock)
    _refresh(tmp_path, vendor, clock)
    clock.advance(days=1)
    vendor.lists["/api/filings/latest"] = [
        filing_row("10-K", PLUG, clock.now().date().isoformat(), (clock.now() - timedelta(hours=2)).isoformat()),
        filing_row("4", TSM, clock.now().date().isoformat()),  # a form no statement endpoint is driven by
    ]
    vendor.calls.clear()
    result = _refresh(tmp_path, vendor, clock)

    assert not result.failed, result.summary()
    assert {params["cik"] for params in _statement_calls(vendor)} == {str(PLUG)}
    assert len(_statement_calls(vendor)) == len(statement_endpoints())  # every statement endpoint, once
    # the filing list follows every form the feed carries: both filers' lists are re-pulled
    assert {params["cik"] for params in vendor.requested("/api/filings") if str(params["page"]) == "1"} == {str(PLUG), str(TSM)}
    # the profile follows any form too; the split history only the reports that tag a split
    assert {params["cik"] for params in vendor.requested("/api/company/details")} == {str(PLUG), str(TSM)}
    assert {params["cik"] for params in vendor.requested("/api/company/stock-splits")} == {str(PLUG)}
    assert raw.read_watermark(tmp_path / "stockfit", "income_statement_quarter", str(TSM)) == (clock.now() - timedelta(days=1)).date()

    clock.advance(days=1)  # the 10-K is still in the 30-day feed, but already re-pulled
    vendor.calls.clear()
    _refresh(tmp_path, vendor, clock)
    assert _statement_calls(vendor) == []
    # a filing with only a date counts from the end of its day, so the pull made that day may miss it:
    # the date-only Form 4 re-pulls its filer once more, the timed 10-K does not
    assert {params["cik"] for params in vendor.requested("/api/filings")} == {str(TSM)}


def test_the_weekly_re_pull_of_the_statements_happens_once_a_week(env, tmp_path, clock):
    vendor = FakeVendor(clock)
    pulled_on = []
    for day in range(15):
        vendor.calls.clear()
        _refresh(tmp_path, vendor, clock)
        if _statement_calls(vendor):
            assert len(_statement_calls(vendor)) == len(statement_endpoints()) * 3  # every company's statements
            pulled_on.append(day)
        clock.advance(days=1)

    assert pulled_on == [0, 7, 14]


def test_running_out_of_the_days_requests_defers_the_rest_without_failing(env, tmp_path, clock):
    from quantlab.dataset.stockfit import raw

    vendor = FakeVendor(clock, per_day=16)
    result = _refresh(tmp_path, vendor, clock)

    assert not result.failed, result.summary()
    assert "deferred" in result.summary()
    assert 429 not in vendor.statuses and len(vendor.calls) <= 16
    assert raw.company_ciks(tmp_path / "stockfit") == [TSM, PLUG, TWTR]


def test_the_endpoint_table_lists_the_market_lists_first_then_the_backfill_order():
    from quantlab.acquisition.stockfit.endpoints import ENDPOINTS, MarketList

    names = list(ENDPOINTS)
    lists = [isinstance(endpoint, MarketList) for endpoint in ENDPOINTS.values()]
    # every company list, feed and forward list before the per-entity endpoints (the funds' own lists aside)
    assert lists.index(False) == names.index("company_details")
    backfill = ["company_details", "income_statement_annual", "footnote_leases_annual", "as_reported_annual",
                "growth_quarter", "insider_transactions", "beneficial_owners_history", "executives_compensation",
                "filings", "fund_profile", "institutional_holders"]
    assert [names.index(name) for name in backfill] == sorted(names.index(name) for name in backfill)


@pytest.mark.parametrize("form", ["10-K/A", "10-Q/A", "40-F"])
def test_an_amendment_or_a_40f_re_pulls_the_filers_statements(env, tmp_path, clock, form):
    vendor = FakeVendor(clock)
    _refresh(tmp_path, vendor, clock)
    clock.advance(days=1)
    accepted = (clock.now() - timedelta(hours=2)).isoformat()
    vendor.lists["/api/filings/latest"] = [filing_row(form, PLUG, clock.now().date().isoformat(), accepted)]
    vendor.calls.clear()
    assert not _refresh(tmp_path, vendor, clock).failed

    assert {params["cik"] for params in _statement_calls(vendor)} == {str(PLUG)}


def test_the_daily_snapshots_start_on_the_first_day_of_a_backfill_longer_than_a_day(env, tmp_path, clock):
    from quantlab.dataset.stockfit import raw

    vendor = FakeVendor(clock, companies=[company_row(GOOGL, "Alphabet Inc.")])
    # every market-wide list, then 10 requests: a fifth of them, 2, for the snapshots
    vendor.per_day = list_requests(all_but_13f()) + 10
    result = _refresh(tmp_path, vendor, clock)

    root = tmp_path / "stockfit"
    assert "deferred" in result.summary()  # the backfill goes on tomorrow
    assert raw.read_pull(root, "filings_calendar", str(GOOGL)) == fixture("filings_calendar_googl")
    assert raw.read_pull(root, "news", str(GOOGL)) == fixture("news_googl")
    assert raw.read_watermark(root, "income_statement_annual", str(GOOGL)) == clock.now().date()  # the backfill too


def test_without_a_snapshot_share_the_snapshots_wait_for_the_backfill(env, tmp_path, clock):
    from quantlab.dataset.stockfit import raw

    vendor = FakeVendor(clock, companies=[company_row(GOOGL, "Alphabet Inc.")])
    vendor.per_day = list_requests(all_but_13f()) + 10
    _refresh(tmp_path, vendor, clock, snapshot_share=0)

    assert raw.entities(tmp_path / "stockfit", "news") == []
