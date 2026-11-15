"""StockFit's forward snapshots: pulled on every refresh, and the panels that accumulate from them.

Seam 1, the raw-tier refresh entry, with ``tests.stockfit_fixtures.FakeVendor``
playing StockFit; seam 2, each panel built from the raw tier on disk. Every
snapshot is placed at the time it was pulled, so a panel shows a value only
from the first snapshot that carried it. List rows are VERBATIM field names
with SYNTHETIC values; recorded bodies are VERBATIM.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.stockfit_fixtures import (
    GOOGL,
    KEY,
    PLUG,
    FakeClock,
    FakeVendor,
    company_row,
    fixture,
)

#: The first refresh runs on Monday 2026-11-16 at 09:00 New York time.
DAYS = ["2026-11-13", "2026-11-16", "2026-11-17", "2026-11-18", "2026-11-19"]


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("STOCKFIT_API_KEY", KEY)


@pytest.fixture
def clock():
    return FakeClock()


def _refresh(tmp_path, vendor, clock, *endpoints, ciks=None):
    from quantlab.acquisition.stockfit.client import StockFitClient
    from quantlab.acquisition.stockfit.refresh import refresh_raw_tier

    client = StockFitClient(http_get=vendor, sleep=clock.sleep, now=clock.now)
    result = refresh_raw_tier(tmp_path, client=client, endpoints=("symbols", "company_delisted", *endpoints), ciks=ciks)
    assert not result.failed, result.summary()
    return result


def _three_refreshes(tmp_path, vendor, clock, *endpoints, between=None, ciks=None):
    """Refresh on Monday, Wednesday and Friday at 09:00 New York; ``between(day)`` edits the vendor before each."""
    for day in range(3):
        if between is not None:
            between(day)
        _refresh(tmp_path, vendor, clock, *endpoints, ciks=ciks)
        clock.advance(days=2)


def _panel(tmp_path, config_name, dataset_path, start="2026-11-13", end="2026-11-19"):
    import importlib

    from quantlab.dataset import config as configs

    module, name = dataset_path.rsplit(".", 1)
    dataset_cls = getattr(importlib.import_module(module), name)
    config = getattr(configs, config_name)(
        zarr_file_path=str(tmp_path / f"{name}.zarr"),
        raw_data_dir_path=str(tmp_path / "stockfit"),
        start_date=start,
        end_date=end,
    )
    dataset_cls(config).update()
    return dataset_cls(config).panel(start, end).load()


def _days(panel):
    return [str(day)[:10] for day in panel["timestamp"].values]


def _values(panel, variable, symbol):
    return [None if isinstance(v, float) and math.isnan(v) else v for v in panel[variable].sel(symbol=symbol).values.tolist()]


# -- listing status ------------------------------------------------------------


def _listing(tmp_path, vendor, clock, between):
    _three_refreshes(tmp_path, vendor, clock, between=between)
    return _panel(tmp_path, "StockFitListingConfig", "quantlab.dataset.stockfit.snapshots.StockFitListingDataset")


def test_a_listing_shows_nothing_before_the_first_company_list_snapshot(env, tmp_path, clock):
    vendor = FakeVendor(clock, companies=[company_row(PLUG, "Plug Power Inc")])

    panel = _listing(tmp_path, vendor, clock, None)

    assert _days(panel) == DAYS
    assert _values(panel, "operating", PLUG) == [None, 1.0, 1.0, 1.0, 1.0]
    names = panel["industry"].attrs["names"]
    assert [names[str(int(code))] if code is not None else None for code in _values(panel, "industry", PLUG)] == [
        None, "Electrical Equipment", "Electrical Equipment", "Electrical Equipment", "Electrical Equipment",
    ]


def test_a_changed_sfic_industry_shows_from_the_day_of_the_snapshot_that_carried_it(env, tmp_path, clock):
    vendor = FakeVendor(clock, companies=[company_row(PLUG, "Plug Power Inc")])

    def between(day):
        if day == 1:
            vendor.lists["/api/symbols"][0]["industry"] = "Hydrogen & Fuel Cells"  # SYNTHETIC

    panel = _listing(tmp_path, vendor, clock, between)

    names = panel["industry"].attrs["names"]
    shown = [names[str(int(code))] if code is not None else None for code in _values(panel, "industry", PLUG)]
    assert shown == [None, "Electrical Equipment", "Electrical Equipment", "Hydrogen & Fuel Cells", "Hydrogen & Fuel Cells"]
    sectors = panel["sector"].attrs["names"]
    assert {sectors[str(int(code))] for code in _values(panel, "sector", PLUG)[1:]} == {"Industrials"}


def test_a_delisting_shows_its_day_and_status_from_the_snapshot_that_first_listed_it_as_delisted(env, tmp_path, clock):
    vendor = FakeVendor(clock, companies=[company_row(PLUG, "Plug Power Inc")])

    def between(day):
        if day == 1:  # SYNTHETIC: delisted on Tuesday, reported by Wednesday's snapshot
            row = vendor.lists["/api/symbols"][0]
            row.update(status="delisted", symbols=[], exchanges=[])
            vendor.lists["/api/company/delisted"] = [dict(row, delistedAt="2026-11-17")]

    panel = _listing(tmp_path, vendor, clock, between)

    assert _values(panel, "operating", PLUG) == [None, 1.0, 1.0, 0.0, 0.0]
    delisted = panel["delisting_day"].sel(symbol=PLUG).values
    assert [str(day)[:10] if not np.isnat(day) else None for day in delisted] == [
        None, None, None, "2026-11-17", "2026-11-17",
    ]


# -- earnings consensus ---------------------------------------------------------


def _upcoming(tickers, earnings, eps=None, analysts=None, report_time=None):
    row = {  # VERBATIM field names, SYNTHETIC values
        "earningsDate": earnings, "filingDate": earnings, "filingType": "10-Q",
        "symbols": tickers, "name": "Some Company",
    }
    if eps is not None:
        row.update(epsEstimate=eps, numEstimates=analysts)
    if report_time is not None:
        row["reportTime"] = report_time
    return row


def _earnings_panel(tmp_path):
    return _panel(
        tmp_path, "StockFitEarningsConsensusConfig",
        "quantlab.dataset.stockfit.snapshots.StockFitEarningsConsensusDataset",
    )


def test_earnings_consensus_shows_each_estimate_from_the_snapshot_that_carried_it_on_the_ticker_holders_cik(
    env, tmp_path, clock
):
    plug = company_row(PLUG, "Plug Power Inc")
    plug["symbols"] = ["PLUG"]
    vendor = FakeVendor(clock, companies=[plug])

    def between(day):
        vendor.lists["/api/earnings/upcoming"] = [
            [_upcoming(["PLUG"], "2026-11-24")],  # predicted: no estimate, no report time
            [_upcoming(["PLUG"], "2026-11-24", eps=-0.12, analysts=9, report_time="after_hours")],
            [_upcoming(["PLUG"], "2026-11-24", eps=-0.10, analysts=10, report_time="after_hours")],
        ][day]

    _three_refreshes(tmp_path, vendor, clock, "earnings_upcoming", between=between)
    panel = _earnings_panel(tmp_path)

    assert _values(panel, "eps_estimate", PLUG) == [None, None, None, -0.12, -0.12]
    assert _values(panel, "analyst_count", PLUG) == [None, None, None, 9.0, 9.0]
    assert _values(panel, "confirmed", PLUG) == [None, 0.0, 0.0, 1.0, 1.0]
    dates = panel["earnings_date"].sel(symbol=PLUG).values
    assert [str(day)[:10] if not np.isnat(day) else None for day in dates] == [None] + ["2026-11-24"] * 4


def test_a_company_the_earnings_calendar_no_longer_lists_shows_nothing_from_that_snapshot_on(env, tmp_path, clock):
    plug = company_row(PLUG, "Plug Power Inc")
    plug["symbols"] = ["PLUG"]
    vendor = FakeVendor(clock, companies=[plug, company_row(GOOGL, "Alphabet Inc.")])

    def between(day):
        rows = [_upcoming(["ALPH"], "2026-12-01", eps=2.0, analysts=30, report_time="after_hours")]
        if day == 0:
            rows.append(_upcoming(["PLUG"], "2026-11-17", eps=-0.12, analysts=9, report_time="pre_market"))
        vendor.lists["/api/earnings/upcoming"] = rows

    _three_refreshes(tmp_path, vendor, clock, "earnings_upcoming", between=between)
    panel = _earnings_panel(tmp_path)

    assert _values(panel, "eps_estimate", PLUG) == [None, -0.12, -0.12, None, None]
    assert _values(panel, "eps_estimate", GOOGL) == [None, 2.0, 2.0, 2.0, 2.0]


# -- IPO calendar and S-1 offerings ---------------------------------------------


def test_an_ipo_deal_shows_on_the_company_holding_its_proposed_ticker_and_moves_stage_with_the_snapshots(
    env, tmp_path, clock
):
    newco = company_row(GOOGL, "Newco Holdings")
    newco["symbols"] = ["NEWC"]
    vendor = FakeVendor(clock, companies=[newco])
    upcoming = fixture("company_ipo_calendar_upcoming")["data"][0]  # VERBATIM row
    unknown = dict(upcoming, proposedTicker="ZZZZ")  # SYNTHETIC: no company holds it

    def between(day):
        deal = dict(upcoming, proposedTicker="NEWC")  # SYNTHETIC ticker
        if day == 0:
            vendor.lists["/api/company/ipo-calendar"] = [deal, unknown]
        else:  # SYNTHETIC: priced on Tuesday
            priced = {k: v for k, v in deal.items() if k not in ("priceRangeLow", "priceRangeHigh", "expectedPriceDate")}
            vendor.lists["/api/company/ipo-calendar"] = [dict(priced, status="priced", price=6.0, pricedDate="2026-11-17")]

    _three_refreshes(tmp_path, vendor, clock, "ipo_calendar", between=between)
    panel = _panel(tmp_path, "StockFitIpoCalendarConfig", "quantlab.dataset.stockfit.snapshots.StockFitIpoCalendarDataset")

    assert list(panel["symbol"].values) == [GOOGL]
    stages = panel["stage"].attrs["names"]
    assert [stages[str(int(s))] if s is not None else None for s in _values(panel, "stage", GOOGL)] == [
        None, "upcoming", "upcoming", "priced", "priced",
    ]
    assert _values(panel, "price_range_high", GOOGL) == [None, 7.0, 7.0, None, None]
    assert _values(panel, "price", GOOGL) == [None, None, None, 6.0, 6.0]


def test_an_s1_offering_shows_on_its_cik_from_the_first_snapshot_listing_it(env, tmp_path, clock):
    vendor = FakeVendor(clock)
    offering = fixture("filings_recent_offerings")["data"][1]  # VERBATIM row, CIK 1854963

    def between(day):
        vendor.lists["/api/filings/recent-offerings"] = [offering] if day >= 1 else []

    _three_refreshes(tmp_path, vendor, clock, "recent_offerings", between=between)
    panel = _panel(tmp_path, "StockFitOfferingsConfig", "quantlab.dataset.stockfit.snapshots.StockFitOfferingsDataset")

    cik = offering["cik"]
    assert _values(panel, "priced", cik) == [None, None, None, 1.0, 1.0]
    assert _values(panel, "price_range_low", cik) == [None, None, None, offering["priceRangeLow"], offering["priceRangeLow"]]
    filed = panel["filed_date"].sel(symbol=cik).values
    assert [str(day)[:10] if not np.isnat(day) else None for day in filed] == [None, None, None, "2026-11-13", "2026-11-13"]


# -- per-company snapshots: filing calendar and news ------------------------------


def _calendar_body(ten_q):
    body = fixture("filings_calendar_googl")  # VERBATIM, with the 10-Q date edited when SYNTHETIC
    for entry in body["nextExpected"]:
        if entry["type"] == "10-Q":
            entry["expectedDate"] = ten_q
    return body


def test_the_refresh_re_pulls_each_operating_companys_snapshots_daily_after_the_backfill(env, tmp_path, clock):
    from quantlab.dataset.stockfit import raw

    vendor = FakeVendor(clock, companies=[company_row(GOOGL, "Alphabet Inc."), company_row(PLUG, "Gone Inc", "delisted")])
    _refresh(tmp_path, vendor, clock, "filings_calendar", "news")
    clock.advance(hours=2)
    _refresh(tmp_path, vendor, clock, "filings_calendar", "news")  # the same day: nothing due
    clock.advance(days=1)
    _refresh(tmp_path, vendor, clock, "filings_calendar", "news")

    root = tmp_path / "stockfit"
    assert len(vendor.requested("/api/filings/calendar")) == 2
    assert [params["cik"] for params in vendor.requested("/api/lookup/news")] == [str(GOOGL)] * 2
    assert raw.read_watermark(root, "news", str(GOOGL)) == clock.now().date()
    assert raw.entities(root, "news") == [str(GOOGL)]


def test_a_changed_predicted_filing_date_shows_from_the_day_of_the_pull_that_carried_it(env, tmp_path, clock):
    vendor = FakeVendor(clock, companies=[company_row(GOOGL, "Alphabet Inc.")])
    key = ("/api/filings/calendar", frozenset({("cik", str(GOOGL))}))

    def between(day):
        vendor.bodies[key] = _calendar_body("2027-02-24" if day == 0 else "2027-02-10")

    _three_refreshes(tmp_path, vendor, clock, "filings_calendar", between=between)
    panel = _panel(
        tmp_path, "StockFitFilingCalendarConfig", "quantlab.dataset.stockfit.snapshots.StockFitFilingCalendarDataset"
    )

    expected = panel["expected_10q"].sel(symbol=GOOGL).values
    assert [str(day)[:10] if not np.isnat(day) else None for day in expected] == [
        None, "2027-02-24", "2027-02-24", "2027-02-10", "2027-02-10",
    ]
    names = panel["confidence_10k"].attrs["names"]
    assert names[str(int(_values(panel, "confidence_10k", GOOGL)[1]))] == "high"


def _headline(title, published):
    return {  # VERBATIM field names, SYNTHETIC values
        "title": title, "link": f"https://example.org/{title}", "source": "example.org",
        "publishedAt": published, "summary": f"Summary of {title}",
    }


def test_a_headline_counts_on_the_later_of_its_publication_and_the_first_pull_carrying_it(env, tmp_path, clock):
    vendor = FakeVendor(clock, companies=[company_row(GOOGL, "Alphabet Inc.")])
    key = ("/api/lookup/news", frozenset({("cik", str(GOOGL))}))
    old = [_headline("old-1", "2026-11-12T15:00:00.000Z"), _headline("old-2", "2026-11-13T15:00:00.000Z")]
    # Published Tuesday after the close, and Wednesday before the morning pull.
    new = [_headline("tue-late", "2026-11-17T22:00:00.000Z"), _headline("wed-early", "2026-11-18T12:00:00.000Z")]

    def between(day):
        vendor.bodies[key] = {"symbol": "GOOGL", "news": (new if day >= 1 else []) + old}

    _three_refreshes(tmp_path, vendor, clock, "news", between=between)
    panel = _panel(tmp_path, "StockFitNewsCountsConfig", "quantlab.dataset.stockfit.snapshots.StockFitNewsCountsDataset")

    assert list(panel.data_vars) == ["news_count"]
    assert panel["news_count"].dtype.kind == "f"
    assert _values(panel, "news_count", GOOGL) == [None, 2.0, 0.0, 2.0, 0.0]


def test_the_recorded_news_body_reaches_the_panel_as_a_count_only(env, tmp_path, clock):
    vendor = FakeVendor(clock, companies=[company_row(GOOGL, "Alphabet Inc.")])
    recorded = fixture("news_googl")["news"]  # VERBATIM: five headlines published by 2026-11-15

    _three_refreshes(tmp_path, vendor, clock, "news")
    panel = _panel(tmp_path, "StockFitNewsCountsConfig", "quantlab.dataset.stockfit.snapshots.StockFitNewsCountsDataset")

    assert _values(panel, "news_count", GOOGL) == [None, float(len(recorded)), 0.0, 0.0, 0.0]
    assert all(panel[name].dtype.kind in "fiubM" for name in panel.data_vars)
    stored = str(panel.attrs) + "".join(str(panel[name].attrs) for name in panel.data_vars)
    assert not any(item["title"] in stored for item in recorded)
