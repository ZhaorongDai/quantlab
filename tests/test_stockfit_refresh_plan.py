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
    KEY,
    PLUG,
    TSM,
    TWTR,
    FakeClock,
    FakeVendor,
    all_but_13f,
    filing_row,
)

STATEMENT_PATHS = ("/api/financials/income-statement", "/api/financials/balance-sheet", "/api/financials/cash-flow-statement")


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("STOCKFIT_API_KEY", KEY)


@pytest.fixture
def clock():
    return FakeClock()


def _refresh(tmp_path, vendor, clock, **kwargs):
    from quantlab.acquisition.stockfit.client import StockFitClient
    from quantlab.acquisition.stockfit.refresh import refresh_raw_tier

    client = StockFitClient(http_get=vendor, sleep=clock.sleep, now=clock.now)
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
    assert not _refresh(uncapped, FakeVendor(FakeClock()), clock).failed

    capped = tmp_path / "capped"
    vendor = FakeVendor(clock, per_day=20)  # 12 for the lists and the feed, 8 for the backfill
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
    assert len(_statement_calls(vendor)) == 9  # three statements, annual, quarterly and TTM
    assert raw.read_watermark(tmp_path / "stockfit", "income_statement_quarter", str(TSM)) == (clock.now() - timedelta(days=1)).date()

    clock.advance(days=1)  # the 10-K is still in the 30-day feed, but already re-pulled
    vendor.calls.clear()
    _refresh(tmp_path, vendor, clock)
    assert _statement_calls(vendor) == []


def test_the_weekly_re_pull_of_the_statements_happens_once_a_week(env, tmp_path, clock):
    vendor = FakeVendor(clock)
    pulled_on = []
    for day in range(15):
        vendor.calls.clear()
        _refresh(tmp_path, vendor, clock)
        if _statement_calls(vendor):
            assert len(_statement_calls(vendor)) == 9 * 3  # every statement of every company
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
