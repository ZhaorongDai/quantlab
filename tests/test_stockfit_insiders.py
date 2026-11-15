"""StockFit's insider transactions in the raw tier, and the insider panel built from them.

Seam 1, the raw-tier refresh entry: ``tests.stockfit_fixtures.FakeVendor``
plays StockFit and serves each company's ``/api/insider-transactions``.
Seam 2: the insider panel is built from that raw tier on disk. The recorded
GOOGL rows are VERBATIM; rows made up for a rule are marked SYNTHETIC.
"""

from __future__ import annotations

from datetime import UTC, datetime

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

#: The Sharadar insider panel's variables, (``SharadarInsidersDataset``).
SHARADAR_INSIDER_VARIABLES = [
    "net_shares", "net_value",
    "net_shares_P", "net_value_P", "net_shares_S", "net_value_S", "net_shares_A", "net_value_A",
    "net_shares_M", "net_value_M", "net_shares_F", "net_value_F", "net_shares_G", "net_value_G",
    "net_shares_D", "net_value_D", "net_shares_other", "net_value_other",
    "net_shares_derivative", "insiders_buying", "insiders_selling",
]


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("STOCKFIT_API_KEY", KEY)


def trade(code, shares, price, accepted, filed, *, side=None, derivative=False, owner="Doe Jane",
          owner_cik=1, traded="2026-11-02", accession="0000000000-26-000001"):
    """One ``/api/insider-transactions`` row in the vendor's shape; SYNTHETIC values."""
    return {
        "ownerName": owner, "ownerCik": owner_cik, "isDirector": False, "isOfficer": True,
        "isTenPercentOwner": False, "officerTitle": "CEO", "documentType": "4",
        "securityTitle": "Option" if derivative else "Common Stock",
        "transactionDate": traded, "transactionCode": code, "shares": shares, "pricePerShare": price,
        "acquiredDisposed": side or ("D" if code == "S" else "A"), "sharesAfter": 1000,
        "directIndirect": "D", "isDerivative": derivative, "exercisePrice": None, "exerciseDate": None,
        "expirationDate": None, "underlyingSecurity": None, "underlyingShares": None,
        "accessionNumber": accession, "filingUrl": "https://www.sec.gov/Archives/edgar/data/1/x.xml",
        "transactionValue": None if price is None else shares * price,
        "filingDate": filed, "acceptedAt": accepted,
    }


def _vendor(clock, trades_by_cik):
    vendor = FakeVendor(clock, companies=[company_row(cik, f"Company {cik}") for cik in trades_by_cik])
    for cik, rows in trades_by_cik.items():
        vendor.bodies[("/api/insider-transactions", frozenset({("cik", str(cik))}))] = {
            "page": 1, "pageSize": 250, "totalPages": 1, "totalResults": len(rows), "data": rows,
        }
    return vendor


def _refresh(download_dir, vendor, clock):
    from quantlab.acquisition.stockfit.refresh import refresh_raw_tier

    client = vendor.client()
    return refresh_raw_tier(download_dir, client=client, endpoints=("symbols", "insider_transactions"))


def _panel(download_dir, start, end):
    from quantlab.dataset.config import StockFitInsidersConfig
    from quantlab.dataset.stockfit.insiders import StockFitInsidersDataset

    def dataset():
        return StockFitInsidersDataset(StockFitInsidersConfig(
            zarr_file_path=str(download_dir / "stockfit_insiders_1d.zarr"),
            raw_data_dir_path=str(download_dir / "stockfit"),
            start_date="2026-01-01",
        ))

    dataset().update()
    return dataset().panel(start, end).load()


def _days_with(panel, variable, symbol):
    series = panel[variable].sel(symbol=symbol)
    return {str(day)[:10]: value for day, value in zip(series["timestamp"].values, series.values.tolist()) if value}


def test_the_refresh_keeps_each_companys_insider_transactions_in_the_raw_tier(env, tmp_path):
    from quantlab.dataset.stockfit import raw

    clock = FakeClock()
    vendor = FakeVendor(clock, companies=[company_row(GOOGL, "Alphabet Inc.")])
    result = _refresh(tmp_path, vendor, clock)

    assert not result.failed, result.summary()
    recorded = fixture("insider_transactions_googl_page1")["data"]
    assert raw.read_pull(tmp_path / "stockfit", "insider_transactions", str(GOOGL)) == recorded
    assert vendor.requested("/api/insider-transactions")[0]["pageSize"] == "250"


def test_the_insider_variables_carry_the_sharadar_panels_names_and_signs(env, tmp_path):
    clock = FakeClock()
    accepted = "2026-11-10T15:00:00Z"  # SYNTHETIC: 10:00 ET on Tuesday
    rows = [  # SYNTHETIC
        trade("P", 100, 10.0, accepted, "2026-11-10", owner="Buyer A", owner_cik=11),
        trade("P", 50, 12.0, accepted, "2026-11-10", owner="Buyer B", owner_cik=12),
        trade("S", 30, 20.0, accepted, "2026-11-10", owner="Seller C", owner_cik=13),
        trade("A", 400, 0.0, accepted, "2026-11-10", owner="Buyer A", owner_cik=11),
        trade("F", 40, 15.0, accepted, "2026-11-10", side="D", owner="Buyer A", owner_cik=11),
        trade("C", 7, 0.0, accepted, "2026-11-10", side="D", owner="Buyer A", owner_cik=11),
        trade("M", 25, 5.0, accepted, "2026-11-10", side="D", derivative=True, owner="Buyer A", owner_cik=11),
    ]
    _refresh(tmp_path, _vendor(clock, {PLUG: rows}), clock)

    panel = _panel(tmp_path, "2026-11-09", "2026-11-11")

    assert sorted(panel.data_vars) == sorted(SHARADAR_INSIDER_VARIABLES)
    day = {name: panel[name].sel(symbol=PLUG).values.tolist() for name in panel.data_vars}
    assert day["net_shares"] == [0.0, 120.0, 0.0]  # 100 + 50 bought, 30 sold
    assert day["net_value"] == [0.0, 1000.0 + 600.0 - 600.0, 0.0]
    assert day["net_shares_P"] == [0.0, 150.0, 0.0]
    assert day["net_shares_S"] == [0.0, -30.0, 0.0]
    assert day["net_shares_A"] == [0.0, 400.0, 0.0]
    assert day["net_shares_F"] == [0.0, -40.0, 0.0]
    assert day["net_value_F"] == [0.0, -600.0, 0.0]
    assert day["net_shares_other"] == [0.0, -7.0, 0.0]
    assert day["net_shares_M"] == [0.0, 0.0, 0.0]  # the exercise is a derivative line
    assert day["net_shares_derivative"] == [0.0, -25.0, 0.0]
    assert day["insiders_buying"] == [0.0, 2.0, 0.0]
    assert day["insiders_selling"] == [0.0, 1.0, 0.0]


@pytest.mark.parametrize(("form", "paths"), [
    ("4", {"/api/insider-transactions"}),
    ("13G", {"/api/ownership/beneficial-owners/history"}),
    ("13D", {"/api/ownership/beneficial-owners/history"}),
    ("DEF 14A", {"/api/executives/compensation", "/api/executives/governance",
                 "/api/executives/performance-measures"}),
])
def test_a_new_insider_13d_13g_or_proxy_filing_re_pulls_only_its_filers_pulls_of_that_kind(env, tmp_path, form, paths):
    from datetime import timedelta

    from quantlab.acquisition.stockfit.refresh import refresh_raw_tier
    from tests.stockfit_fixtures import filing_row

    kinds = {
        "/api/insider-transactions", "/api/ownership/beneficial-owners/history",
        "/api/executives/compensation", "/api/executives/governance", "/api/executives/performance-measures",
    }
    clock = FakeClock()
    vendor = FakeVendor(clock)
    client = vendor.client()
    refresh_raw_tier(tmp_path, client=client)
    clock.advance(days=1)
    accepted = (clock.now() - timedelta(hours=2)).isoformat()
    vendor.lists["/api/filings/latest"] = [filing_row(form, PLUG, clock.now().date().isoformat(), accepted)]
    vendor.calls.clear()

    result = refresh_raw_tier(tmp_path, client=client)

    assert not result.failed, result.summary()
    for path in kinds:
        assert {params["cik"] for params in vendor.requested(path)} == ({str(PLUG)} if path in paths else set()), path


def test_a_line_lands_on_its_filings_available_day_never_its_transaction_day(env, tmp_path):
    clock = FakeClock(datetime(2026, 11, 20, 14, tzinfo=UTC))
    rows = [  # SYNTHETIC: both traded on Monday 2026-11-02
        trade("P", 100, 10.0, "2026-11-10T20:59:00Z", "2026-11-10", owner_cik=21, accession="a-1"),  # 15:59 ET
        trade("S", 10, 30.0, "2026-11-10T21:00:00Z", "2026-11-10", owner_cik=22, accession="a-2"),  # 16:00 ET
    ]
    _refresh(tmp_path, _vendor(clock, {PLUG: rows}), clock)

    panel = _panel(tmp_path, "2026-11-02", "2026-11-13")

    assert _days_with(panel, "net_shares_P", PLUG) == {"2026-11-10": 100.0}
    assert _days_with(panel, "net_shares_S", PLUG) == {"2026-11-11": -10.0}


def test_a_line_an_amendment_repeats_is_counted_on_its_first_filing_only(env, tmp_path):
    clock = FakeClock(datetime(2026, 11, 20, 14, tzinfo=UTC))
    rows = [  # SYNTHETIC
        trade("P", 100, 10.0, "2026-11-10T15:00:00Z", "2026-11-10", accession="b-1"),
        trade("P", 100, 10.0, "2026-11-12T15:00:00Z", "2026-11-12", accession="b-2"),  # a 4/A repeating it
        trade("P", 120, 10.0, "2026-11-12T15:00:00Z", "2026-11-12", accession="b-2"),  # a changed line counts
    ]
    _refresh(tmp_path, _vendor(clock, {PLUG: rows}), clock)

    panel = _panel(tmp_path, "2026-11-09", "2026-11-13")

    assert _days_with(panel, "net_shares", PLUG) == {"2026-11-10": 100.0, "2026-11-12": 120.0}
    assert _days_with(panel, "net_value_P", PLUG) == {"2026-11-10": 1000.0, "2026-11-12": 1200.0}
