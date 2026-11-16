"""StockFit's quarterly and TTM statement panels, as known on each day.

Seam 2 of the StockFit spec, building a panel from a raw tier on disk: the
raw tier is written by the refresh entry from ``FakeVendor``, which replays
the recorded responses (VERBATIM); a test that needs a filing the recordings
lack (an amendment, an acceptance time just before the close) edits the
vendor's body first, and marks the edit SYNTHETIC.

PLUG's quarterly 10-Qs are all accepted after the close. Its 2026-09-30
quarter was accepted 2026-11-09T22:01:51Z, 17:01 in New York; the quarter
before it, 2026-06-30, on 2026-08-10. The vendor's clock pulls on
2026-11-16, so the raw tier is complete through 2026-11-15.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.stockfit_fixtures import KEY, PLUG, TWTR, FakeClock, FakeVendor, fixture

QUARTER_ENDPOINTS = ("income_statement_quarter", "balance_sheet_quarter", "cash_flow_statement_quarter")
INCOME = "/api/financials/income-statement"


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("STOCKFIT_API_KEY", KEY)


def _raw_tier(tmp_path, vendor=None):
    from quantlab.acquisition.stockfit.refresh import refresh_raw_tier

    vendor = vendor if vendor is not None else FakeVendor(FakeClock())
    client = vendor.client()
    result = refresh_raw_tier(tmp_path / "downloads", endpoints=QUARTER_ENDPOINTS, ciks=(PLUG, TWTR), client=client)
    assert not result.failed, result.summary()
    return tmp_path / "downloads" / "stockfit"


def _panel(tmp_path, vendor_root, *, period="quarter", reported="as_known", start="2026-01-01"):
    from quantlab.dataset.config import StockFitStatementsConfig
    from quantlab.dataset.stockfit.statements import StockFitStatementsDataset

    config = StockFitStatementsConfig(
        zarr_file_path=str(tmp_path / f"statements_{period}_{reported}.zarr"),
        raw_data_dir_path=str(vendor_root),
        period=period,
        reported=reported,
        start_date=start,
    )
    dataset = StockFitStatementsDataset(config).update()
    return dataset.panel(start, "2026-12-31").load()


def _on(panel, day, name, cik=PLUG):
    return panel[name].sel(timestamp=pd.Timestamp(day), symbol=cik).item()


def _plug_income(vendor):
    """Return the vendor's PLUG quarterly income statement, to edit in place."""
    (key,) = [k for k in vendor.bodies if k[0] == INCOME and ("cik", str(PLUG)) in k[1] and ("period", "quarter") in k[1]]
    return vendor.bodies[key]


def test_a_quarter_accepted_after_the_close_shows_from_the_next_trading_day(env, tmp_path):
    panel = _panel(tmp_path, _raw_tier(tmp_path))

    assert _on(panel, "2026-11-09", "period_end") == pd.Timestamp("2026-06-30").value
    assert _on(panel, "2026-11-10", "period_end") == pd.Timestamp("2026-09-30").value
    assert _on(panel, "2026-11-10", "revenue") == 200_602_000
    assert _on(panel, "2026-11-09", "revenue") == fixture("income_statement_plug_quarter")[1]["facts"]["revenue"]
    assert panel["symbol"].values.tolist() == [PLUG, TWTR]
    assert panel["timestamp"].values[-1] == np.datetime64("2026-11-13")  # the last session through 2026-11-15


@pytest.mark.parametrize(
    ("accepted_at", "first_day"),
    [
        ("2026-11-09T21:30:00Z", "2026-11-10"),  # SYNTHETIC: 16:30 in New York, after the close
        ("2026-11-09T20:59:00Z", "2026-11-09"),  # SYNTHETIC: 15:59, before it
        (None, "2026-11-09"),  # SYNTHETIC: no acceptance time, so its filing date
    ],
)
def test_a_quarter_is_visible_on_its_filing_day_only_when_accepted_before_the_close(env, tmp_path, accepted_at, first_day):
    vendor = FakeVendor(FakeClock())
    latest = _plug_income(vendor)[0]  # the 2026-09-30 quarter, filed 2026-11-09
    for dated in (latest, latest["sources"]["0001104659-26-124578"]):  # the quarter and its 10-Q
        if accepted_at is None:
            dated.pop("acceptedAt")
        else:
            dated["acceptedAt"] = accepted_at
    panel = _panel(tmp_path, _raw_tier(tmp_path, vendor))

    shown = panel["revenue"].sel(symbol=PLUG).to_series()
    assert shown[shown == 200_602_000].index.min() == pd.Timestamp(first_day)


def _restated(vendor):
    """Make the vendor serve PLUG's 2026-09-30 revenue as restated by a 10-Q/A on 2026-11-12, 10:00 in New York."""
    latest = _plug_income(vendor)[0]
    latest["facts"]["revenue"] = 210_000_000.0  # SYNTHETIC
    latest["sources"]["0001104659-26-990001"] = {  # SYNTHETIC accession
        "type": "10-Q/A",
        "dateFiled": "2026-11-12",
        "acceptedAt": "2026-11-12T15:00:00Z",
        "amendment": True,
        "facts": {"revenue": {"before": 200_602_000}},
    }
    return vendor


def test_a_restated_fact_shows_its_earlier_value_until_the_restatement_is_available(env, tmp_path):
    panel = _panel(tmp_path, _raw_tier(tmp_path, _restated(FakeVendor(FakeClock()))))

    assert _on(panel, "2026-11-10", "revenue") == 200_602_000
    assert _on(panel, "2026-11-11", "revenue") == 200_602_000
    assert _on(panel, "2026-11-12", "revenue") == 210_000_000
    assert _on(panel, "2026-11-12", "costOfRevenue") == 205_960_000  # unrestated facts keep their value
    assert _on(panel, "2026-11-12", "release_date") == pd.Timestamp("2026-11-09").value


def test_the_originally_reported_set_never_changes_after_its_first_filing(env, tmp_path):
    panel = _panel(tmp_path, _raw_tier(tmp_path, _restated(FakeVendor(FakeClock()))), reported="original")

    shown = panel["revenue"].sel(symbol=PLUG, timestamp=slice("2026-11-10", None)).to_series()
    assert set(shown) == {200_602_000}


def test_the_ttm_panel_agrees_with_the_vendors_own_ttm_on_its_latest_quarter(env, tmp_path):
    panel = _panel(tmp_path, _raw_tier(tmp_path), period="ttm")
    vendor_ttm = fixture("income_statement_plug_ttm")[0]["facts"]  # the vendor's TTM to 2026-09-30, VERBATIM

    # Flows summed, share counts averaged, balance-sheet items and closing cash from the latest quarter.
    for name in ("revenue", "netIncome", "eps", "operatingCashFlow", "weightedAverageSharesOutstanding",
                 "assets", "stockholdersEquity", "endCash"):
        assert _on(panel, "2026-11-13", name) == pytest.approx(vendor_ttm[name]), name
    assert _on(panel, "2026-11-13", "period_end") == pd.Timestamp("2026-09-30").value
    # The 2026-09-30 10-Q reports no impairments: the vendor sums the other three quarters, the panel does not.
    assert vendor_ttm["impairments"] == 19_365_000 + 3_856_000 + 724_944_000
    assert np.isnan(_on(panel, "2026-11-13", "impairments"))


def test_a_ttm_value_changes_when_a_quarter_in_it_is_restated(env, tmp_path):
    panel = _panel(tmp_path, _raw_tier(tmp_path, _restated(FakeVendor(FakeClock()))), period="ttm")

    before = _on(panel, "2026-11-11", "revenue")
    assert _on(panel, "2026-11-12", "revenue") - before == 210_000_000 - 200_602_000


def test_a_ttm_flow_one_of_the_four_quarters_lacks_is_missing_not_a_partial_sum(env, tmp_path):
    vendor = FakeVendor(FakeClock())
    (quarter,) = [entry for entry in _plug_income(vendor) if entry["period"] == "2026-06-30"]
    quarter["facts"]["revenue"] = None  # SYNTHETIC: the 10-Q of 2026-06-30 reports no revenue
    panel = _panel(tmp_path, _raw_tier(tmp_path, vendor), period="ttm")

    assert np.isnan(_on(panel, "2026-11-13", "revenue"))
    assert not np.isnan(_on(panel, "2026-11-13", "netIncome"))


def _plug_balance_sheet(vendor):
    """Return the vendor's PLUG quarterly balance sheet, to edit in place."""
    (key,) = [k for k in vendor.bodies
              if k[0] == "/api/financials/balance-sheet" and ("cik", str(PLUG)) in k[1] and ("period", "quarter") in k[1]]
    return vendor.bodies[key]


def test_a_fiscal_year_end_balance_sheet_shows_from_its_10k_though_a_later_10q_repeats_it(env, tmp_path):
    # VERBATIM: the vendor lists PLUG's 2025-12-31 assets under the 10-Q of 2026-08-10 (a later filing that
    # repeats the year-end balance sheet unchanged), not under the 10-K of 2026-03-02 that first reported it.
    (quarter,) = [entry for entry in fixture("balance_sheet_plug_quarter") if entry["period"] == "2025-12-31"]
    assert "assets" not in quarter["sources"]["0001104659-26-022286"]["facts"]
    assert quarter["sources"]["0001104659-26-093454"]["facts"]["assets"] == {}
    panel = _panel(tmp_path, _raw_tier(tmp_path))

    # The 10-K was accepted 2026-03-02 16:30 in New York, after the close.
    assert _on(panel, "2026-03-03", "period_end") == pd.Timestamp("2025-12-31").value
    assert _on(panel, "2026-03-03", "assets") == quarter["facts"]["assets"]
    assert _on(panel, "2026-05-11", "assets") == quarter["facts"]["assets"]


def test_a_cover_page_fact_shows_from_the_filing_that_reports_it_not_the_quarters_own(env, tmp_path):
    vendor = FakeVendor(FakeClock())
    (quarter,) = [entry for entry in _plug_balance_sheet(vendor) if entry["period"] == "2026-06-30"]
    quarter["facts"]["publicFloat"] = 1_500_000_000.0  # SYNTHETIC: a later 10-K's cover page reports the Q2 float
    quarter["sources"]["0001104659-26-990002"] = {  # SYNTHETIC accession
        "type": "10-K", "dateFiled": "2026-11-12", "acceptedAt": "2026-11-12T15:00:00Z", "amendment": False,
        "facts": {"publicFloat": {}},
    }
    panel = _panel(tmp_path, _raw_tier(tmp_path, vendor))

    assert np.isnan(_on(panel, "2026-08-11", "publicFloat"))
    assert np.isnan(_on(panel, "2026-11-09", "publicFloat"))


def test_a_quarter_without_its_own_filing_date_is_dated_by_its_earliest_dated_filing(env, tmp_path):
    vendor = FakeVendor(FakeClock())
    for body in (_plug_income(vendor), _plug_balance_sheet(vendor)):
        (quarter,) = [entry for entry in body if entry["period"] == "2026-06-30"]
        quarter.pop("dateFiled")  # SYNTHETIC: a comparative period the vendor gives no filing of its own
        quarter.pop("acceptedAt")
    panel = _panel(tmp_path, _raw_tier(tmp_path, vendor))

    # Its 10-Q of 2026-08-10, accepted after the close, carries the quarter.
    assert _on(panel, "2026-08-10", "period_end") == pd.Timestamp("2026-03-31").value
    assert _on(panel, "2026-08-11", "period_end") == pd.Timestamp("2026-06-30").value
    (quarter,) = [entry for entry in fixture("balance_sheet_plug_quarter") if entry["period"] == "2026-06-30"]
    assert _on(panel, "2026-08-11", "assets") == quarter["facts"]["assets"]
    assert _on(panel, "2026-08-11", "release_date") == pd.Timestamp("2026-08-10").value


def test_annual_statements_stay_in_the_raw_tier(tmp_path):
    with pytest.raises(ValueError, match="Annual statements stay in the raw tier"):
        _panel(tmp_path, tmp_path, period="annual")


def test_a_quarter_in_several_statements_shows_the_earliest_of_their_filing_dates_as_its_release_date(env, tmp_path):
    vendor = FakeVendor(FakeClock())
    (quarter,) = [entry for entry in _plug_income(vendor) if entry["period"] == "2026-06-30"]
    quarter["dateFiled"] = "2026-11-12"  # SYNTHETIC: the income statement dates the quarter at a later filing
    quarter["acceptedAt"] = "2026-11-12T15:00:00Z"
    panel = _panel(tmp_path, _raw_tier(tmp_path, vendor))

    # The balance sheet and cash-flow statement date it at its 10-Q of 2026-08-10, accepted after the close.
    assert _on(panel, "2026-08-11", "period_end") == pd.Timestamp("2026-06-30").value
    assert _on(panel, "2026-08-11", "release_date") == pd.Timestamp("2026-08-10").value


def _listed_only_later(statement: str, listed: str) -> list:
    """Return the versions of one fact whose only listing is a filing on ``listed``; the period was filed 2016-03-15."""
    from quantlab.dataset.stockfit.statements import fact_versions

    period = {  # SYNTHETIC values and accession, VERBATIM field names
        "dateFiled": "2016-03-15", "acceptedAt": "2016-03-15T21:00:00Z", "facts": {"stockholdersEquity": 7.0},
        "sources": {"0001157601-19-000001": {"type": "10-K", "dateFiled": listed, "facts": {"stockholdersEquity": {}}}},
    }
    return [(version.value, version.date_filed) for version in fact_versions(period, statement)]


@pytest.mark.parametrize(
    ("statement", "listed", "dated"),
    [
        ("balance_sheet", "2017-04-19", "2016-03-15"),  # 400 days: still the next year's comparative column
        ("balance_sheet", "2017-04-20", "2017-04-20"),  # 401 days: too late to be a comparative
        ("income_statement", "2018-04-14", "2016-03-15"),  # 760 days: a 10-K lists three years
        ("income_statement", "2018-04-15", "2018-04-15"),
        ("cash_flow_statement", "2018-04-15", "2018-04-15"),
    ],
)
def test_a_value_first_listed_too_late_to_be_a_comparative_is_dated_at_that_listing(statement, listed, dated):
    assert _listed_only_later(statement, listed) == [(7.0, dated)]
