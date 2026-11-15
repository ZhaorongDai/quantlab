"""StockFit's 13D/13G filings in the raw tier, and the stake panel built from them.

Seam 1, the raw-tier refresh entry: ``tests.stockfit_fixtures.FakeVendor``
plays StockFit and serves PLUG's recorded ``/api/ownership/beneficial-owners/history``
(VERBATIM) and a filing list (``/api/filings``) whose acceptance times are
VERBATIM where open_facts recorded them and SYNTHETIC otherwise. Seam 2: the
panel is built from that raw tier on disk.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest

from tests.stockfit_fixtures import KEY, PLUG, FakeClock, FakeVendor, company_row, fixture


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("STOCKFIT_API_KEY", KEY)


def filing(form, accession, filed, accepted):
    """One ``/api/filings`` row in the vendor's shape."""
    return {
        "type": form, "accessionNumber": accession, "url": "https://www.sec.gov/Archives/edgar/data/1093691/x.xml",
        "dateFiled": filed, "acceptedAt": accepted, "xbrl": False, "amendment": False,
    }


#: PLUG's filing list: the 13D/13G accessions of the recorded history.
PLUG_FILINGS = [
    # VERBATIM acceptance time (open_facts): reportDate "2026-03-28 05:19:13.361", a Saturday.
    filing("13G", "0000102909-26-002096", "2026-03-27", "2026-03-27T15:22:40Z"),
    # VERBATIM: reportDate "2025-07-24 15:11:41" is the acceptance time, 11:11 ET.
    filing("13G", "0001374170-25-000063", "2025-07-24", "2025-07-24T15:11:41Z"),
    # SYNTHETIC: accepted at 16:30 ET on the reportDate, so available the next day.
    filing("13G", "0002012383-26-002858", "2026-07-29", "2026-07-29T20:30:00Z"),
    # SYNTHETIC: the joint 13D of five reporting persons, accepted 20:59 ET the evening before its reportDate.
    filing("13D", "0001829655-25-000009", "2025-06-25", "2025-06-26T00:59:45Z"),
    # Vanguard Capital Management's 0002100119-26-000952 is not in the list: it is never placed.
]


def _refresh(download_dir, clock):
    from quantlab.acquisition.stockfit.client import StockFitClient
    from quantlab.acquisition.stockfit.refresh import refresh_raw_tier

    vendor = FakeVendor(clock, companies=[company_row(PLUG, "Plug Power Inc")])
    vendor.bodies[("/api/filings", frozenset({("cik", str(PLUG))}))] = {
        "page": 1, "pageSize": 100, "totalPages": 1, "totalResults": len(PLUG_FILINGS), "data": PLUG_FILINGS,
    }
    client = StockFitClient(http_get=vendor, sleep=clock.sleep, now=clock.now)
    result = refresh_raw_tier(
        download_dir, client=client, endpoints=("symbols", "beneficial_owners_history", "filings"),
    )
    assert not result.failed, result.summary()
    return vendor


def _panel(download_dir, start, end):
    from quantlab.dataset.config import StockFitBeneficialOwnersConfig
    from quantlab.dataset.stockfit.beneficial_owners import StockFitBeneficialOwnersDataset

    def dataset():
        return StockFitBeneficialOwnersDataset(StockFitBeneficialOwnersConfig(
            zarr_file_path=str(download_dir / "stockfit_beneficial_owners_1d.zarr"),
            raw_data_dir_path=str(download_dir / "stockfit"),
            start_date="2025-01-01",
        ))

    dataset().update()
    return dataset().panel(start, end).load()


def _days_with(panel, variable):
    series = panel[variable].sel(symbol=PLUG)
    return {
        str(day)[:10]: value
        for day, value in zip(series["timestamp"].values, series.values.tolist())
        if value and not (isinstance(value, float) and math.isnan(value))
    }


def test_the_refresh_keeps_each_companys_13d_13g_filings_in_the_raw_tier(env, tmp_path):
    from quantlab.dataset.stockfit import raw

    _refresh(tmp_path, FakeClock(datetime(2026, 11, 16, 14, tzinfo=UTC)))

    recorded = fixture("beneficial_owners_history_plug")
    assert raw.read_pull(tmp_path / "stockfit", "beneficial_owners_history", str(PLUG)) == recorded


def test_a_13d_13g_filing_is_placed_by_its_acceptance_time_never_by_report_date(env, tmp_path):
    _refresh(tmp_path, FakeClock(datetime(2026, 11, 16, 14, tzinfo=UTC)))

    panel = _panel(tmp_path, "2025-01-02", "2026-11-13")

    assert _days_with(panel, "filings_13g") == {
        "2025-07-24": 1,  # accepted 11:11 ET
        "2026-03-27": 1,  # accepted Friday 11:22 ET; reportDate would say Saturday, so Monday
        "2026-07-30": 1,  # accepted 16:30 ET on its reportDate
    }
    # Five reporting persons share one accession: one filing, on the day after its 20:59 ET acceptance.
    assert _days_with(panel, "filings_13d") == {"2025-06-26": 1}
    assert _days_with(panel, "percent_of_class") == {"2025-06-26": 4.8, "2025-07-24": 0.4, "2026-07-30": 12.8}
    assert panel["percent_of_class"].sel(symbol=PLUG, timestamp="2026-03-27").item() == 0.0  # an exit
    assert _days_with(panel, "shares_owned")["2026-07-30"] == 178091159.0
    assert _days_with(panel, "activist") == {}
