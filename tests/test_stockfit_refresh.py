"""StockFit's raw-tier refresh: company reference and the three statements.

Seam 1 of the StockFit spec, the raw-tier refresh entry: HTTP is replaced by
``tests.stockfit_fixtures.FakeVendor``, which replays the recorded responses
and keeps the vendor's rate limits. What lands on disk is read back only
through ``quantlab.dataset.stockfit.raw``, the module that owns the layout.
"""

from __future__ import annotations

import json

import pytest
from loguru import logger

from tests.stockfit_fixtures import (
    GOOGL,
    KEY,
    PLUG,
    TSM,
    TWTR,
    FakeClock,
    FakeVendor,
    all_but_13f,
    company_row,
    fixture,
)

STATEMENTS = ("income_statement", "balance_sheet", "cash_flow_statement")


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("STOCKFIT_API_KEY", KEY)


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def log_lines():
    lines: list[str] = []
    sink = logger.add(lambda message: lines.append(str(message)), level="DEBUG")
    yield lines
    logger.remove(sink)


def _client(vendor, clock, **kwargs):
    from quantlab.acquisition.stockfit.client import StockFitClient

    return StockFitClient(http_get=vendor, sleep=clock.sleep, now=clock.now, **kwargs)


def _refresh(tmp_path, vendor, clock, **kwargs):
    from quantlab.acquisition.stockfit.refresh import refresh_raw_tier

    kwargs.setdefault("endpoints", all_but_13f())
    return refresh_raw_tier(tmp_path, client=_client(vendor, clock), **kwargs)


def test_a_refresh_lands_the_company_lists_and_each_companys_statements(env, tmp_path, clock):
    from quantlab.dataset.stockfit import raw

    result = _refresh(tmp_path, FakeVendor(clock), clock)

    root = tmp_path / "stockfit"
    assert not result.failed, result.summary()
    assert raw.company_ciks(root) == [TSM, PLUG, TWTR]
    assert [row["cik"] for row in raw.read_pull(root, "symbols", "delisted")] == [TWTR]
    assert raw.read_pull(root, "income_statement_quarter", str(PLUG)) == fixture("income_statement_plug_quarter")
    assert raw.read_pull(root, "balance_sheet_ttm", str(PLUG)) == fixture("balance_sheet_plug_ttm")
    assert raw.read_pull(root, "cash_flow_statement_annual", str(TSM)) == fixture("cash_flow_statement_tsm_annual")
    assert len(raw.pulls(root, "income_statement_quarter", str(PLUG))) == 1
    assert raw.read_watermark(root, "income_statement_quarter", str(PLUG)) == clock.now().date()


def test_an_unchanged_re_pull_publishes_nothing_and_only_moves_the_watermark(env, tmp_path, clock):
    from quantlab.dataset.stockfit import raw

    vendor = FakeVendor(clock)
    _refresh(tmp_path, vendor, clock)
    clock.advance(days=7)  # the weekly re-pull of the statements
    result = _refresh(tmp_path, vendor, clock)

    root = tmp_path / "stockfit"
    assert "unchanged" in result.summary() and "published" not in result.summary()
    assert len(raw.pulls(root, "income_statement_quarter", str(PLUG))) == 1
    assert len(raw.pulls(root, "symbols", "operating")) == 1
    assert raw.read_watermark(root, "income_statement_quarter", str(PLUG)) == clock.now().date()


def test_a_changed_re_pull_publishes_a_new_pull_and_keeps_the_old(env, tmp_path, clock):
    from quantlab.dataset.stockfit import raw

    vendor = FakeVendor(clock)
    _refresh(tmp_path, vendor, clock)
    (first,) = raw.pulls(tmp_path / "stockfit", "income_statement_quarter", str(PLUG))
    restated = fixture("income_statement_plug_quarter")
    restated[0]["facts"]["revenue"] += 1  # SYNTHETIC restatement
    key = ("/api/financials/income-statement", frozenset({("cik", str(PLUG)), ("period", "quarter")}))
    vendor.bodies[key] = restated
    vendor.lists["/api/symbols"].append(company_row(GOOGL, "Alphabet Inc."))
    clock.advance(days=7)  # the weekly re-pull of the statements
    _refresh(tmp_path, vendor, clock)

    root = tmp_path / "stockfit"
    original = fixture("income_statement_plug_quarter")
    (kept, repulled) = raw.pulls(root, "income_statement_quarter", str(PLUG))  # the original and the weekly re-pull
    assert kept == first and repulled.date() == clock.now().date()
    assert raw.read_pull(root, "income_statement_quarter", str(PLUG), first) == original
    assert raw.read_pull(root, "income_statement_quarter", str(PLUG)) == restated
    assert raw.read_pull_as_of(root, "income_statement_quarter", str(PLUG), first) == original
    assert len(raw.pulls(root, "income_statement_annual", str(TSM))) == 1
    assert len(raw.pulls(root, "symbols", "operating")) == 2
    assert GOOGL in raw.company_ciks(root)


def test_a_killed_refresh_leaves_no_partial_pull_and_the_next_one_publishes_it_whole(env, tmp_path, clock):
    from dataclasses import replace

    from quantlab.acquisition.stockfit.endpoints import ENDPOINTS
    from quantlab.dataset.stockfit import raw

    companies = [company_row(1000 + i, f"Company {i}") for i in range(5)]  # SYNTHETIC
    small_pages = replace(ENDPOINTS["symbols"], page_size=2)
    vendor = FakeVendor(clock, companies=companies)
    calls = []

    def dies_on_the_second_page(url, *, params, headers):
        calls.append(params)
        if len(calls) == 2:
            raise KeyboardInterrupt  # the process is killed mid-pull
        return vendor(url, params=params, headers=headers)

    with pytest.raises(KeyboardInterrupt):
        _refresh(tmp_path, dies_on_the_second_page, clock, endpoints=[small_pages])
    root = tmp_path / "stockfit"
    assert raw.pulls(root, "symbols", "operating") == []
    assert raw.company_ciks(root) == []
    assert raw.read_watermark(root, "symbols", "operating") is None

    _refresh(tmp_path, vendor, clock, endpoints=[small_pages])
    assert [row["cik"] for row in raw.read_pull(root, "symbols", "operating")] == [1000, 1001, 1002, 1003, 1004]
    assert not raw.pending_dir(root, "symbols", "operating").exists()


def test_the_refresh_keeps_inside_the_per_minute_limit_the_headers_state(env, tmp_path, clock):
    vendor = FakeVendor(clock, per_minute=4)
    lists_and_statements = ["symbols", "company_delisted"] + [
        f"{name}_{period}" for period in ("annual", "quarter", "ttm") for name in STATEMENTS
    ]
    result = _refresh(tmp_path, vendor, clock, endpoints=lists_and_statements)

    assert not result.failed, result.summary()
    assert len(vendor.calls) == 3 + 9 * 3  # three list pages, nine statements of three companies
    assert 429 not in vendor.statuses
    assert clock.sleeps and sum(clock.sleeps) >= 60 * (len(vendor.calls) // 4 - 1)


def test_the_refresh_stops_before_the_daily_cap_and_sends_nothing_past_it(env, tmp_path, clock):
    from quantlab.acquisition.stockfit.refresh import refresh_raw_tier
    from quantlab.dataset.stockfit import raw

    vendor = FakeVendor(clock, per_day=5)
    client = _client(vendor, clock, day_reserve=1)
    result = refresh_raw_tier(tmp_path, client=client)

    assert len(vendor.calls) == 4 and 429 not in vendor.statuses
    assert not result.failed and "deferred to the next run" in result.summary()
    root = tmp_path / "stockfit"
    assert raw.company_ciks(root) == [TSM, PLUG, TWTR]
    assert raw.entities(root, "income_statement_annual") == []


def test_the_refresh_refuses_without_the_key_before_any_request(tmp_path, monkeypatch):
    from quantlab.acquisition.stockfit.client import StockFitCredentialError
    from quantlab.acquisition.stockfit.refresh import refresh_raw_tier

    monkeypatch.delenv("STOCKFIT_API_KEY", raising=False)
    with pytest.raises(StockFitCredentialError, match="STOCKFIT_API_KEY"):
        refresh_raw_tier(tmp_path)
    assert not (tmp_path / "stockfit").exists()


def test_an_endpoint_stockfit_refuses_fails_its_part_and_the_others_still_run(env, tmp_path, clock):
    from quantlab.dataset.stockfit import raw

    vendor = FakeVendor(clock, replies=[(403, b'{"error":"Feature not available on current plan"}')])
    result = _refresh(
        tmp_path, vendor, clock, endpoints=["income_statement_quarter", "balance_sheet_quarter"], ciks=[PLUG, TWTR]
    )

    assert result.failed == ("income_statement_quarter",)
    assert "403" in result.summary()
    assert raw.entities(tmp_path / "stockfit", "balance_sheet_quarter") == [str(PLUG), str(TWTR)]


def test_stockfit_is_a_registered_source_refreshed_through_the_registry(env, tmp_path, clock, monkeypatch):
    from quantlab.acquisition import registry
    from quantlab.acquisition.base import DataSourceRegistry
    from quantlab.acquisition.stockfit import client as client_module
    from quantlab.dataset.stockfit import raw

    source = DataSourceRegistry.get("stockfit")
    assert source.required_env == (client_module.API_KEY_ENV,)
    assert KEY not in repr(source)
    assert source.symbol_batch_cls is None and source.config_factory is None

    from quantlab.acquisition.stockfit import refresh as refresh_module

    vendor = FakeVendor(clock)
    monkeypatch.setattr(refresh_module, "StockFitClient", lambda: _client(vendor, clock))
    result = registry.refresh_raw_tier(source, tmp_path)
    assert result.vendor == "stockfit" and not result.failed, result.summary()
    assert raw.company_ciks(tmp_path / "stockfit") == [TSM, PLUG, TWTR]


def test_the_key_appears_in_no_file_url_query_or_log_line(env, tmp_path, clock, log_lines):
    vendor = FakeVendor(clock, per_minute=10, replies=[(500, b"oops")])
    _refresh(tmp_path, vendor, clock)

    assert all(headers == {"Authorization": f"Bearer {KEY}"} for _, _, headers in vendor.calls)
    assert all(KEY not in url + json.dumps(params) for url, params, _ in vendor.calls)
    assert log_lines and all(KEY not in line for line in log_lines)
    files = [path for path in (tmp_path / "stockfit").rglob("*") if path.is_file()]
    assert files and all(KEY not in path.read_text() for path in files)


def test_the_data_dictionaries_land_raw_and_an_unchanged_list_is_dropped(env, tmp_path, clock):
    from quantlab.dataset.stockfit import raw

    vendor = FakeVendor(clock, companies=[company_row(GOOGL, "Alphabet Inc.")])
    names = ["symbols", "symbol_sectors", "symbol_industries", "symbol_exchanges", "executives_officers"]
    assert not _refresh(tmp_path, vendor, clock, endpoints=names).failed
    clock.advance(days=1)
    assert not _refresh(tmp_path, vendor, clock, endpoints=names).failed

    root = tmp_path / "stockfit"
    assert raw.read_pull(root, "symbol_sectors", "all") == fixture("symbols_sectors")
    assert raw.read_pull(root, "symbol_industries", "all") == fixture("symbols_industries")
    assert raw.read_pull(root, "symbol_exchanges", "all") == fixture("symbols_exchanges")
    assert raw.read_pull(root, "executives_officers", str(GOOGL)) == fixture("executives_officers_googl")
    # the lists are pulled on every run, the unchanged second pull dropped; the officers wait 30 days
    assert len(raw.pulls(root, "symbol_sectors", "all")) == 1
    assert raw.read_watermark(root, "symbol_sectors", "all") == clock.now().date()
    assert len(vendor.requested("/api/executives/officers")) == 1
