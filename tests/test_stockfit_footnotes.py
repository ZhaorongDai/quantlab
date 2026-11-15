"""StockFit's footnote tables: the raw pulls and one panel per footnote.

Seam 1 is the raw-tier refresh entry (``refresh_raw_tier`` over
``tests.stockfit_fixtures.FakeVendor``); seam 2 is a footnote panel built
from that raw tier on disk (``StockFitFootnoteDataset``). The recorded
footnote bodies are VERBATIM StockFit responses for Alphabet, Boeing and
Realty Income; edits to them are marked SYNTHETIC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from datetime import timedelta

from tests.stockfit_fixtures import GOOGL, KEY, FakeClock, FakeVendor, filing_row, fixture

BA, O = 12927, 726728
COMPANIES = (GOOGL, BA, O)


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("STOCKFIT_API_KEY", KEY)


@pytest.fixture
def clock():
    return FakeClock()


def _refresh(tmp_path, vendor, clock, ciks=COMPANIES):
    from quantlab.acquisition.stockfit.client import StockFitClient
    from quantlab.acquisition.stockfit.refresh import refresh_raw_tier
    from quantlab.dataset.stockfit.footnotes import FOOTNOTES, footnote_endpoint

    footnotes = [footnote_endpoint(name, period) for name in FOOTNOTES for period in ("annual", "quarter")]
    endpoints = ["filings_latest", *footnotes]  # the feed whose new reports re-pull the footnotes
    client = StockFitClient(http_get=vendor, sleep=clock.sleep, now=clock.now)
    result = refresh_raw_tier(tmp_path, client=client, endpoints=endpoints, ciks=ciks)
    assert not result.failed, result.summary()
    return result


def _file_report(vendor, clock, cik=GOOGL, form="10-Q"):
    """Make the recent-filings feed serve a report the company filed an hour ago: its footnotes are re-pulled."""
    filed = clock.now() - timedelta(hours=1)
    vendor.lists["/api/filings/latest"] = [filing_row(form, cik, filed.date().isoformat(), filed.isoformat())]  # SYNTHETIC


def _next_day_refresh(tmp_path, vendor, clock):
    """Re-pull the next day, nothing new served: the raw tier is complete through the day before."""
    clock.advance(days=1)
    _file_report(vendor, clock)
    _refresh(tmp_path, vendor, clock, ciks=(GOOGL,))


def _dataset(tmp_path, footnote, period="annual", start="2014-01-01", **kwargs):
    from quantlab.dataset.config import StockFitFootnoteConfig
    from quantlab.dataset.stockfit.footnotes import StockFitFootnoteDataset

    return StockFitFootnoteDataset(StockFitFootnoteConfig(
        zarr_file_path=str(tmp_path / "stores" / f"{footnote}_{period}.zarr"),
        raw_data_dir_path=str(tmp_path / "stockfit"),
        footnote=footnote,
        period=period,
        start_date=start,
        **kwargs,
    ))


def _panel(tmp_path, footnote, period="annual", **kwargs):
    dataset = _dataset(tmp_path, footnote, period, **kwargs)
    dataset.update()
    return _dataset(tmp_path, footnote, period, **kwargs).panel("2014-01-01", "2026-12-31").load()


def _on(panel, variable, day, cik=GOOGL):
    return panel[variable].sel(timestamp=pd.Timestamp(day), symbol=cik).item()


def test_a_footnote_value_is_shown_from_its_filing_date_on(env, tmp_path, clock):
    _refresh(tmp_path, FakeVendor(clock), clock)
    panel = _panel(tmp_path, "backlog", start="2024-06-01")

    rpo = "remaining_performance_obligation"
    assert np.isnan(_on(panel, rpo, "2025-02-04"))
    assert _on(panel, rpo, "2025-02-05") == 93_200_000_000
    assert _on(panel, rpo, "2026-02-04") == 93_200_000_000
    assert _on(panel, rpo, "2026-02-05") == 242_800_000_000
    assert _on(panel, "period_end", "2026-02-05") == pd.Timestamp("2025-12-31").value
    assert panel["symbol"].values.tolist() == [GOOGL]


# One value per footnote, read off the recorded body by hand: (company, variable, day, value).
SHOWN = {
    "revenue_segmentation": (GOOGL, "us_revenue", "2026-02-05", 194_229_000_000),
    "business_segmentation": (GOOGL, "period_end", "2026-10-29", pd.Timestamp("2025-12-31").value),
    "debt_structure": (GOOGL, "period_end", "2026-10-29", pd.Timestamp("2025-12-31").value),
    "credit_facilities": (GOOGL, "period_end", "2026-10-29", pd.Timestamp("2025-12-31").value),
    "leases": (GOOGL, "operating_right_of_use_asset", "2026-02-05", 15_221_000_000),
    "stock_compensation": (GOOGL, "expense", "2026-07-23", 27_100_000_000),
    "concentration": (GOOGL, "geographic_revenue_share", "2026-02-05", 1.0),
    "retirement_plans": (BA, "pension_benefit_obligation", "2026-01-30", 51_425_000_000),
    "restructuring": (GOOGL, "severance_charges", "2024-01-31", 2_100_000_000),
    "asset_composition": (GOOGL, "us_long_lived_assets", "2026-10-29", 195_337_000_000),
    "buybacks": (GOOGL, "repurchased_value", "2026-02-05", 45_398_000_000),
    "subsidiary_financials": (GOOGL, "subsidiaries_revenue", "2016-02-11", 74_989_000_000),
    "fair_value_hierarchy": (GOOGL, "assets_level1", "2026-10-29", 15_751_000_000),
    "reit_properties": (O, "period_end", "2019-03-01", pd.Timestamp("2018-12-31").value),
    "backlog": (GOOGL, "expected_share_in_window", "2026-02-05", np.nan),
    "supplier_finance": (BA, "obligation", "2026-04-22", 1_994_000_000),
    "workforce_cost": (GOOGL, "accrued_compensation", "2026-02-05", 17_546_000_000),
}


@pytest.fixture(scope="module")
def footnote_tier(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("footnotes")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("STOCKFIT_API_KEY", KEY)
        clock = FakeClock()
        _refresh(tmp_path, FakeVendor(clock), clock)
    return tmp_path


def test_every_footnote_has_a_spec_and_is_pulled_annual_and_quarterly():
    from quantlab.acquisition.stockfit.endpoints import ENDPOINTS
    from quantlab.dataset.stockfit.footnotes import FOOTNOTES

    assert sorted(FOOTNOTES) == sorted(SHOWN) and len(FOOTNOTES) == 17
    pulled = {endpoint.path for endpoint in ENDPOINTS.values() if endpoint.path.startswith("/api/footnotes/")}
    assert pulled == {spec.path for spec in FOOTNOTES.values()}
    assert ENDPOINTS["footnote_leases_quarter"].query("1652044") == {"period": "quarter", "limit": 80, "cik": 1652044}


@pytest.mark.parametrize("footnote", sorted(SHOWN))
def test_each_footnote_builds_a_panel_from_the_recorded_bodies(footnote_tier, footnote):
    from quantlab.dataset.stockfit.footnotes import COMMON_VARIABLES, FOOTNOTES

    cik, variable, day, value = SHOWN[footnote]
    panel = _panel(footnote_tier, footnote)

    assert set(panel.data_vars) == {*COMMON_VARIABLES, *FOOTNOTES[footnote].fields}
    assert cik in panel["symbol"].values.tolist()
    np.testing.assert_equal(_on(panel, variable, day, cik), value)
    assert _on(panel, "may_include_restatement", day, cik) == 1.0


def test_per_member_lists_stay_in_the_raw_tier(footnote_tier):
    from quantlab.dataset.stockfit import raw

    root = footnote_tier / "stockfit"
    body = raw.read_pull(root, "footnote_debt_structure_annual", str(GOOGL))
    assert body == fixture("footnote_debt_structure_googl")
    assert len(body[0]["instruments"]) == 16
    panel = _panel(footnote_tier, "debt_structure")
    assert not any("instrument" in name or "tranche" in name for name in panel.data_vars)


def test_a_quarterly_panel_reads_the_quarterly_pulls(footnote_tier):
    panel = _panel(footnote_tier, "buybacks", "quarter")

    assert _on(panel, "cash_spent", "2026-10-29") == 0
    assert _on(panel, "period_end", "2026-10-29") == pd.Timestamp("2026-09-30").value
    assert _on(panel, "period_end", "2026-07-23") == pd.Timestamp("2026-06-30").value


# -- restatements: flagged before the first pull, point in time after it -------------

BACKLOG = ("/api/footnotes/backlog", frozenset({("cik", str(GOOGL)), ("period", "annual")}))
RPO = "remaining_performance_obligation"


def test_a_change_a_re_pull_sees_is_shown_from_the_pull_on_and_not_flagged(env, tmp_path, clock):
    vendor = FakeVendor(clock)
    _refresh(tmp_path, vendor, clock, ciks=(GOOGL,))
    restated = fixture("footnote_backlog_googl")
    restated[0]["remainingPerformanceObligation"] = 250_000_000_000  # SYNTHETIC silent restatement
    vendor.bodies[BACKLOG] = restated
    clock.advance(days=2)  # Wednesday 2026-11-18, 09:00 in New York
    _file_report(vendor, clock)
    _refresh(tmp_path, vendor, clock, ciks=(GOOGL,))
    _next_day_refresh(tmp_path, vendor, clock)
    panel = _panel(tmp_path, "backlog", start="2024-06-01")

    assert _on(panel, RPO, "2026-11-17") == 242_800_000_000
    assert _on(panel, "may_include_restatement", "2026-11-17") == 1.0
    assert _on(panel, RPO, "2026-11-18") == 250_000_000_000
    assert _on(panel, "may_include_restatement", "2026-11-18") == 0.0
    assert _on(panel, "filing_date", "2026-11-18") == pd.Timestamp("2026-02-05").value


def test_a_change_pulled_after_the_close_is_shown_from_the_next_trading_day(env, tmp_path, clock):
    vendor = FakeVendor(clock)
    _refresh(tmp_path, vendor, clock, ciks=(GOOGL,))
    restated = fixture("footnote_backlog_googl")
    restated[0]["remainingPerformanceObligation"] = 250_000_000_000  # SYNTHETIC
    vendor.bodies[BACKLOG] = restated
    clock.advance(days=4, hours=8)  # Friday 2026-11-20, 17:00 in New York
    _file_report(vendor, clock)
    _refresh(tmp_path, vendor, clock, ciks=(GOOGL,))
    clock.advance(days=4)  # Tuesday: the raw tier is complete through Monday
    _file_report(vendor, clock)
    _refresh(tmp_path, vendor, clock, ciks=(GOOGL,))
    panel = _panel(tmp_path, "backlog", start="2024-06-01")

    assert _on(panel, RPO, "2026-11-20") == 242_800_000_000
    assert _on(panel, RPO, "2026-11-23") == 250_000_000_000


def test_a_period_filed_after_the_first_pull_is_shown_from_its_filing_date_unflagged(env, tmp_path):
    clock = FakeClock(pd.Timestamp("2026-01-05 14:00", tz="UTC").to_pydatetime())
    vendor = FakeVendor(clock)
    vendor.bodies[BACKLOG] = fixture("footnote_backlog_googl")[1:]  # SYNTHETIC: before the FY2025 10-K
    _refresh(tmp_path, vendor, clock, ciks=(GOOGL,))
    vendor.bodies[BACKLOG] = fixture("footnote_backlog_googl")
    clock.advance(days=56)  # 2026-03-02
    _file_report(vendor, clock, form="10-K")
    _refresh(tmp_path, vendor, clock, ciks=(GOOGL,))
    panel = _panel(tmp_path, "backlog", start="2024-06-01")

    assert _on(panel, RPO, "2026-02-04") == 93_200_000_000
    assert _on(panel, "may_include_restatement", "2026-02-04") == 1.0
    assert _on(panel, RPO, "2026-02-05") == 242_800_000_000
    assert _on(panel, "may_include_restatement", "2026-02-05") == 0.0


def test_a_restated_older_period_never_replaces_the_newer_one_shown(env, tmp_path, clock):
    vendor = FakeVendor(clock)
    _refresh(tmp_path, vendor, clock, ciks=(GOOGL,))
    restated = fixture("footnote_backlog_googl")
    restated[1]["remainingPerformanceObligation"] = 1.0  # SYNTHETIC restatement of FY2024
    vendor.bodies[BACKLOG] = restated
    clock.advance(days=2)
    _file_report(vendor, clock)
    _refresh(tmp_path, vendor, clock, ciks=(GOOGL,))
    _next_day_refresh(tmp_path, vendor, clock)
    panel = _panel(tmp_path, "backlog", start="2024-06-01")

    assert _on(panel, RPO, "2026-11-18") == 242_800_000_000
    assert _on(panel, "period_end", "2026-11-18") == pd.Timestamp("2025-12-31").value


# -- the vendor's rollups, and the specs as data --------------------------------------


def test_the_vendors_debt_type_totals_and_level3_share_are_shown(env, tmp_path, clock):
    vendor = FakeVendor(clock)
    debt = fixture("footnote_debt_structure_googl")
    debt[0]["types"] = [  # SYNTHETIC rollups in the vendor's shape
        {"member": "us-gaap:SeniorNotesMember", "name": "Senior Notes", "currency": "USD",
         "faceAmount": 30_000_000_000, "carryingAmount": 29_500_000_000},
        {"member": "us-gaap:CommercialPaperMember", "name": "Commercial Paper", "currency": "USD",
         "faceAmount": 2_000_000_000},
    ]
    vendor.bodies[("/api/footnotes/debt-structure", BACKLOG[1])] = debt
    fair_value = fixture("footnote_fair_value_hierarchy_googl")
    fair_value[0]["measures"][0].update(level3=1_000_000_000, level3Share=0.0089)  # SYNTHETIC
    vendor.bodies[("/api/footnotes/fair-value-hierarchy", BACKLOG[1])] = fair_value
    _refresh(tmp_path, vendor, clock, ciks=(GOOGL,))

    debt_panel = _panel(tmp_path, "debt_structure", start="2024-06-01")
    assert _on(debt_panel, "senior_notes_face_amount", "2026-10-29") == 30_000_000_000
    assert _on(debt_panel, "senior_notes_carrying_amount", "2026-10-29") == 29_500_000_000
    assert _on(debt_panel, "commercial_paper_face_amount", "2026-10-29") == 2_000_000_000
    assert np.isnan(_on(debt_panel, "commercial_paper_carrying_amount", "2026-10-29"))
    fair_value_panel = _panel(tmp_path, "fair_value_hierarchy", start="2024-06-01")
    assert _on(fair_value_panel, "assets_level3_share", "2026-10-29") == 0.0089
    assert _on(fair_value_panel, "assets_level3", "2026-10-29") == 1_000_000_000
    assert _on(fair_value_panel, "investments_level2", "2026-10-29") == 91_733_000_000


def test_adding_a_field_to_a_footnote_is_one_line_of_its_spec(env, tmp_path, clock, monkeypatch):
    from dataclasses import replace

    from quantlab.dataset.stockfit.footnotes import FOOTNOTES

    spec = FOOTNOTES["leases"]
    added = replace(spec, fields={**spec.fields, "operating_payments_year1_again": "[].operating.payments.year1"})
    monkeypatch.setitem(FOOTNOTES, "leases", added)
    _refresh(tmp_path, FakeVendor(clock), clock, ciks=(GOOGL,))
    panel = _panel(tmp_path, "leases", start="2024-06-01")

    assert _on(panel, "operating_payments_year1_again", "2026-02-05") == 3_275_000_000


def test_an_unknown_footnote_or_period_is_refused(tmp_path):
    with pytest.raises(ValueError, match="unknown StockFit footnote 'pensions'"):
        _dataset(tmp_path, "pensions")
    with pytest.raises(ValueError, match="period must be one of"):
        _dataset(tmp_path, "leases", "ttm")
