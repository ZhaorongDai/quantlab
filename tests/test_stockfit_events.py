"""StockFit corporate event panels: historical splits, successions, declared splits and dividends.

Seam 2 of the StockFit spec, a panel built from a raw tier on disk: the raw
tier is written by the refresh entry over ``tests.stockfit_fixtures.FakeVendor``,
then the panel is built from it. No event here adjusts a price.
"""

from __future__ import annotations

import pandas as pd
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

SPLITS = ("/api/company/stock-splits", frozenset({("cik", str(GOOGL))}))
DETAILS = ("/api/company/details", frozenset({("cik", str(GOOGL))}))
UPCOMING_SPLITS = "/api/company/upcoming-stock-splits"
UPCOMING_DIVIDENDS = "/api/earnings/upcoming-dividends"
EVENT_ENDPOINTS = ("symbols", "company_details", "stock_splits", "upcoming_stock_splits", "upcoming_dividends")


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("STOCKFIT_API_KEY", KEY)


@pytest.fixture
def clock():
    return FakeClock()  # 2026-11-16 14:00 UTC, a Monday morning in New York


def _companies():
    return [  # SYNTHETIC rows; GOOGL's tickers are its real ones
        dict(company_row(GOOGL, "Alphabet Inc."), symbols=["GOOGL", "GOOG"]),
        dict(company_row(PLUG, "Plug Power Inc"), symbols=["PLUG"]),
    ]


def _vendor(clock, *, splits=(), dividends=()):
    vendor = FakeVendor(clock, companies=_companies())
    vendor.lists[UPCOMING_SPLITS] = list(splits)
    vendor.lists[UPCOMING_DIVIDENDS] = list(dividends)
    return vendor


def _refresh(tmp_path, vendor, clock):
    from quantlab.acquisition.stockfit.refresh import refresh_raw_tier

    client = vendor.client()
    result = refresh_raw_tier(tmp_path, client=client, endpoints=EVENT_ENDPOINTS)
    assert not result.failed, result.summary()


def _corporate(tmp_path, start="2024-01-01"):
    from quantlab.dataset.config import StockFitCorporateEventsConfig
    from quantlab.dataset.stockfit.events import StockFitCorporateEventsDataset

    return StockFitCorporateEventsDataset(StockFitCorporateEventsConfig(
        zarr_file_path=str(tmp_path / "corporate.zarr"),
        raw_data_dir_path=str(tmp_path / "stockfit"),
        start_date=start,
    ))


def _declared(tmp_path, start="2026-11-02"):
    from quantlab.dataset.config import StockFitDeclaredEventsConfig
    from quantlab.dataset.stockfit.events import StockFitDeclaredEventsDataset

    return StockFitDeclaredEventsDataset(StockFitDeclaredEventsConfig(
        zarr_file_path=str(tmp_path / "declared.zarr"),
        raw_data_dir_path=str(tmp_path / "stockfit"),
        start_date=start,
    ))


def _shown(panel, variable, symbol):
    """Return ``{day: value}`` of the days a symbol shows a value of ``variable``."""
    series = panel[variable].sel(symbol=symbol).to_series().dropna()
    return {day.strftime("%Y-%m-%d"): value for day, value in series.items()}


def test_a_split_disclosed_60_days_after_its_date_appears_on_its_disclosure_day(env, tmp_path, clock):
    vendor = _vendor(clock)
    vendor.bodies[SPLITS] = [  # SYNTHETIC: a 3-for-1 split on 2024-03-01, first tagged 60 days later
        {"date": "2024-03-01", "ratio": 3, "type": "forward", "accession": "0001652044-24-000001",
         "firstDisclosedAt": "2024-04-30"},
    ]
    _refresh(tmp_path, vendor, clock)

    panel = _corporate(tmp_path).update().panel("2024-01-01", "2024-12-31").load()

    assert _shown(panel, "split_ratio", GOOGL) == {"2024-04-30": 3.0}
    assert _shown(panel, "split_date", GOOGL) == {"2024-04-30": pd.Timestamp("2024-03-01")}


def test_a_succession_shows_on_both_companies_on_its_effective_day(env, tmp_path, clock):
    google_inc = 1288776
    vendor = _vendor(clock)
    vendor.lists["/api/symbols"].append(company_row(google_inc, "GOOGLE INC.", "delisted"))
    link = {"relationType": "successor", "effectiveDate": "2015-10-02", "corroborated": True}  # SYNTHETIC
    vendor.bodies[DETAILS] = dict(fixture("company_details_googl"),
                                  predecessors=[dict(link, name="GOOGLE INC.", cik=google_inc)])
    google_key = ("/api/company/details", frozenset({("cik", str(google_inc))}))
    vendor.bodies[google_key] = dict(fixture("company_details_google_inc"),
                                     successors=[dict(link, name="Alphabet Inc.", cik=GOOGL)])
    _refresh(tmp_path, vendor, clock)

    panel = _corporate(tmp_path, start="2015-01-02").update().panel("2015-01-02", "2015-12-31").load()

    assert _shown(panel, "predecessor_cik", GOOGL) == {"2015-10-02": float(google_inc)}
    assert _shown(panel, "successor_cik", google_inc) == {"2015-10-02": float(GOOGL)}
    assert _shown(panel, "split_ratio", GOOGL) == {}


def test_the_recorded_split_history_is_shown_on_disclosure_days_only(env, tmp_path, clock):
    _refresh(tmp_path, _vendor(clock), clock)

    panel = _corporate(tmp_path, start="2014-01-02").update().panel("2014-01-02", "2022-12-30").load()

    # VERBATIM: the 2014 split was first tagged on 2016-02-11, the 2022 one before it happened.
    assert _shown(panel, "split_ratio", GOOGL) == {"2016-02-11": 2.0, "2022-04-27": 20.0}
    assert panel["split_ratio"].sel(symbol=GOOGL, timestamp="2014-04-02").isnull()


def _declared_split(tickers, day, ratio):
    return {"symbols": tickers, "name": "x", "splitDate": day, "ratio": ratio,  # SYNTHETIC
            "ratioText": f"{ratio}:1", "type": "forward"}


def _declared_dividend(tickers, ex, amount):
    return {"symbols": tickers, "name": "x", "exDate": ex, "paymentDate": None, "recordDate": None,  # SYNTHETIC
            "announcementDate": None, "dividendPerShare": amount, "indicatedAnnualDividend": amount * 4,
            "currency": "USD"}


def test_a_declared_split_shows_from_its_snapshots_day_through_its_split_date(env, tmp_path, clock):
    vendor = _vendor(clock, splits=[_declared_split(["GOOG"], "2026-11-18", 2.0)])
    _refresh(tmp_path, vendor, clock)  # Monday 2026-11-16, 09:00 New York
    vendor.lists[UPCOMING_SPLITS] = []  # the split executed; the next calendar no longer has it
    clock.advance(days=3)
    _refresh(tmp_path, vendor, clock)  # Thursday 2026-11-19

    panel = _declared(tmp_path).update().panel("2026-11-02", "2026-11-19").load()

    assert _shown(panel, "declared_split_ratio", GOOGL) == {"2026-11-16": 2.0, "2026-11-17": 2.0, "2026-11-18": 2.0}
    assert _shown(panel, "declared_split_date", GOOGL)["2026-11-17"] == pd.Timestamp("2026-11-18")


def test_a_snapshot_pulled_after_the_close_shows_from_the_next_trading_day(env, tmp_path, clock):
    clock.advance(hours=7)  # 21:00 UTC is 16:00 in New York: the session has closed
    vendor = _vendor(clock, dividends=[_declared_dividend(["PLUG"], "2026-11-20", 0.25)])
    _refresh(tmp_path, vendor, clock)
    clock.advance(days=2)
    _refresh(tmp_path, vendor, clock)  # Wednesday: the raw tier is complete through Tuesday

    panel = _declared(tmp_path).update().panel("2026-11-02", "2026-11-17").load()

    assert _shown(panel, "declared_dividend", PLUG) == {"2026-11-17": 0.25}
    assert _shown(panel, "declared_dividend_annual", PLUG) == {"2026-11-17": 1.0}


def test_declared_rows_whose_tickers_name_no_company_are_left_out(env, tmp_path, clock):
    vendor = _vendor(clock, dividends=[
        _declared_dividend(["GOOGL", "GOOG"], "2026-11-17", 0.21),
        _declared_dividend(["GOOGL"], "2026-11-17", 0.01),
        _declared_dividend(["ZZZZ"], "2026-11-17", 9.0),
    ])
    _refresh(tmp_path, vendor, clock)
    clock.advance(days=1)
    _refresh(tmp_path, vendor, clock)  # the raw tier is complete through the first refresh's day

    panel = _declared(tmp_path).update().panel("2026-11-02", "2026-11-17").load()

    assert panel["symbol"].values.tolist() == [GOOGL]
    assert _shown(panel, "declared_dividend", GOOGL) == pytest.approx({"2026-11-16": 0.22})


def test_the_declared_calendars_are_snapshotted_on_every_refresh_that_changes_them(env, tmp_path, clock):
    from quantlab.dataset.stockfit import raw

    vendor = _vendor(clock, splits=fixture("upcoming_stock_splits_page1")["data"])
    _refresh(tmp_path, vendor, clock)
    first = clock.now()
    clock.advance(days=1)
    _refresh(tmp_path, vendor, clock)  # unchanged: dropped
    vendor.lists[UPCOMING_SPLITS] = vendor.lists[UPCOMING_SPLITS][1:]
    clock.advance(days=1)
    _refresh(tmp_path, vendor, clock)

    root = tmp_path / "stockfit"
    assert raw.pulls(root, "upcoming_stock_splits", "all") == [first, clock.now()]
    assert raw.read_pull(root, "upcoming_stock_splits", "all", first) == fixture("upcoming_stock_splits_page1")["data"]
    assert str(vendor.requested(UPCOMING_SPLITS)[0]["days"]) == "28"
    assert raw.read_watermark(root, "upcoming_dividends", "all") == clock.now().date()
