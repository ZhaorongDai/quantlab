"""StockFit's proxy executive data in the raw tier, and the executive panel built from it.

Seam 1, the raw-tier refresh entry: ``tests.stockfit_fixtures.FakeVendor``
plays StockFit and serves GOOGL's recorded compensation, governance and
performance measures (VERBATIM). The annual income statements, which give
each fiscal year's end, and the second company are SYNTHETIC. Seam 2: the
panel is built from that raw tier on disk.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest

from tests.stockfit_fixtures import GOOGL, KEY, FakeClock, FakeVendor, company_row

#: A SYNTHETIC company whose fiscal year ends in September.
SEPTEMBER = 320193


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("STOCKFIT_API_KEY", KEY)


def annual(period, fiscal_year):
    """One annual income-statement period in the vendor's shape; SYNTHETIC values."""
    return {"period": period, "fiscalYear": fiscal_year, "fiscalPeriod": "FY", "dateFiled": None,
            "acceptedAt": None, "facts": {"revenue": 1.0}, "sources": {}}


def _refresh(download_dir, clock):
    from quantlab.acquisition.stockfit.client import StockFitClient
    from quantlab.acquisition.stockfit.refresh import refresh_raw_tier

    vendor = FakeVendor(clock, companies=[company_row(GOOGL, "Alphabet Inc."), company_row(SEPTEMBER, "September Co")])
    statement = ("/api/financials/income-statement", "annual")
    vendor.bodies[(statement[0], frozenset({("cik", str(GOOGL)), ("period", "annual")}))] = [
        annual(f"{year}-12-31", year) for year in range(2025, 2019, -1)
    ]
    vendor.bodies[(statement[0], frozenset({("cik", str(SEPTEMBER)), ("period", "annual")}))] = [
        annual("2025-09-27", 2025),
    ]
    vendor.bodies[("/api/executives/compensation", frozenset({("cik", str(SEPTEMBER))}))] = [{
        "fiscalYear": 2025, "ceoTotalComp": 74000000, "ceoActuallyPaidComp": 168000000,
        "avgOfficerTotalComp": 27000000, "avgOfficerActuallyPaidComp": 60000000,
        "companyTsr": 250.0, "peerGroupTsr": 200.0,
    }]
    client = StockFitClient(http_get=vendor, sleep=clock.sleep, now=clock.now)
    result = refresh_raw_tier(download_dir, client=client, endpoints=(
        "symbols", "income_statement_annual",
        "executives_compensation", "executives_governance", "executives_performance_measures",
    ))
    assert not result.failed, result.summary()


def _panel(download_dir, start, end):
    from quantlab.dataset.config import StockFitExecutivesConfig
    from quantlab.dataset.stockfit.executives import StockFitExecutivesDataset

    def dataset():
        return StockFitExecutivesDataset(StockFitExecutivesConfig(
            zarr_file_path=str(download_dir / "stockfit_executives_1d.zarr"),
            raw_data_dir_path=str(download_dir / "stockfit"),
            start_date="2021-01-01",
        ))

    dataset().update()
    return dataset().panel(start, end).load()


def _values(panel, variable, symbol, *days):
    return [
        None if math.isnan(value) else value
        for value in (panel[variable].sel(symbol=symbol, timestamp=day).item() for day in days)
    ]


def test_executive_values_for_a_fiscal_year_appear_120_days_after_its_end(env, tmp_path):
    _refresh(tmp_path, FakeClock(datetime(2026, 11, 16, 14, tzinfo=UTC)))

    panel = _panel(tmp_path, "2025-01-02", "2026-11-13")

    # FY2024 ended 2024-12-31: available 2025-04-30; FY2025 on 2026-04-30 (VERBATIM values).
    days = ("2025-04-29", "2025-04-30", "2026-04-29", "2026-04-30")
    assert _values(panel, "ceo_total_comp", GOOGL, *days) == [8802824.0, 10800694.0, 10800694.0, 10906079.0]
    assert _values(panel, "company_tsr", GOOGL, *days) == [160.89, 218.2, 218.2, 360.95]
    assert _values(panel, "company_selected_measure", GOOGL, *days) == [87.0, 73.0, 73.0, 0.95]
    # A fiscal year ended 2025-09-27: 120 days later is Sunday 2026-01-25, so Monday.
    assert _values(panel, "ceo_total_comp", SEPTEMBER, "2026-01-23", "2026-01-26") == [None, 74000000.0]


def _later_unchanged_pulls(download_dir, at):
    """Move every executive endpoint's watermark to ``at``: re-pulls that served nothing new."""
    from quantlab.acquisition.stockfit.raw_tier import write_entity_watermark
    from quantlab.dataset.stockfit.raw import entities

    root = download_dir / "stockfit"
    for endpoint in ("income_statement_annual", "executives_compensation", "executives_governance",
                     "executives_performance_measures"):
        for entity in entities(root, endpoint):
            write_entity_watermark(root, endpoint, entity, at)


def test_governance_flags_belong_to_the_newest_fiscal_year_known_when_pulled(env, tmp_path):
    _refresh(tmp_path, FakeClock(datetime(2026, 11, 16, 14, tzinfo=UTC)))
    _later_unchanged_pulls(tmp_path, datetime(2026, 11, 20, 14, tzinfo=UTC))

    panel = _panel(tmp_path, "2025-01-02", "2026-11-19")

    # The flags belong to FY2025 (available 2026-04-30), but were first pulled on 2026-11-16 (before the
    # close): a pull's flags are never shown before its day, while FY2025's pay shows from 2026-04-30.
    days = ("2026-04-30", "2026-11-13", "2026-11-16")
    assert _values(panel, "insider_trading_policy", GOOGL, *days) == [None, None, 1.0]
    assert _values(panel, "mnpi_disclosure_timed_for_comp", GOOGL, *days) == [None, None, 0.0]
    assert _values(panel, "ceo_total_comp", GOOGL, *days) == [10906079.0] * 3
