"""StockFit's fund series: the raw-tier pulls and the fund panels.

Seam 1, the raw-tier refresh entry, with ``tests.stockfit_fixtures.FakeVendor``
replaying the recorded Vanguard 500 Index Fund (VOO, trust CIK 36405, series
2839) responses; seam 2, a fund panel built from the raw tier that refresh
left on disk. A second series of the same trust is SYNTHETIC.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from tests.stockfit_fixtures import KEY, FakeClock, FakeVendor, company_row, fixture

TRUST = 36405
VOO = f"{TRUST}-2839"
OTHER = f"{TRUST}-2840"  # SYNTHETIC second series of the same trust

#: The fund endpoints a fund refresh pulls, in pull order.
FUND_ENDPOINTS = (
    "fund_symbols", "fund_daily_supported", "fund_profile", "fund_flows", "fund_fees", "fund_performance",
    "fund_holdings", "fund_holdings_daily", "fund_health", "fund_fee_analysis", "fund_service_providers",
    "fund_by_adviser",
)


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("STOCKFIT_API_KEY", KEY)


@pytest.fixture
def clock():
    return FakeClock()


def _fund_row(series_id: int, symbols: list[str], kind: str) -> dict:
    row = company_row(TRUST, "500 Index Fund", kind=kind)  # SYNTHETIC values, VERBATIM field names
    return dict(row, seriesId=series_id, symbols=symbols, sector=None, industry=None)


def _one_page(body: dict) -> dict:
    return dict(copy.deepcopy(body), totalPages=1)


def _key(path: str, **params) -> tuple[str, frozenset]:
    return path, frozenset((k, str(v)) for k, v in params.items())


def _vendor(clock) -> FakeVendor:
    """The recorded VOO responses, one page each, plus a SYNTHETIC second series of the trust."""
    vendor = FakeVendor(clock, companies=[
        company_row(1093691, "Plug Power Inc"),
        _fund_row(2839, ["VOO"], "mf"),
        _fund_row(2840, ["VXX0"], "etf"),
    ])
    for name, path in (("fund_fees_voo", "/api/fund/fees"), ("fund_performance_voo", "/api/fund/performance"),
                       ("fund_holdings_daily_voo_page1", "/api/fund/holdings/daily")):
        vendor.bodies[_key(path, symbol="VOO")] = _one_page(fixture(name))
    holdings = _one_page(fixture("fund_holdings_voo_page1"))
    vendor.bodies[_key("/api/fund/holdings", symbol="VOO", reportDate="2026-06-30")] = holdings
    vendor.bodies[("/api/fund/holdings/daily/supported-funds", frozenset())] = ["IVV", "VOO"]
    other = copy.deepcopy(fixture("fund_profile_voo")[:2])
    for row in other:  # SYNTHETIC: the second series is a tenth of the first
        row.update(seriesId=2840, seriesName="SYNTHETIC SECOND SERIES", netAssets=row["netAssets"] / 10)
    vendor.bodies[_key("/api/fund/profile", symbol="VXX0")] = other
    vendor.bodies[_key("/api/fund/fund-health", symbol="VOO")] = FUND_HEALTH
    vendor.bodies[_key("/api/fund/fee-analysis", symbol="VOO")] = FEE_ANALYSIS
    vendor.bodies[_key("/api/fund/service-providers", symbol="VOO")] = SERVICE_PROVIDERS
    vendor.bodies[_key("/api/fund/by-adviser", crd="105958")] = BY_ADVISER
    return vendor


# SYNTHETIC N-CEN bodies in the shapes of StockFit's OpenAPI schemas.
FUND_HEALTH = {
    "reportDate": "2026-06-30",
    "ncenRecords": [
        {"reportDate": "2025-12-31", "isIndexFund": True, "indexReturnDiffBefore": -0.01, "indexReturnDiffAfter": 0.02,
         "indexStdevBefore": 0.01, "indexStdevAfter": 0.01, "isLendingAuthorized": True, "didLend": False,
         "isLendingLiquidated": False, "hasMaterialWeakness": False, "hasLegalProceedings": False,
         "hasValuationChanges": False, "hasAccountingChanges": False, "hasPaymentErrors": False},
    ],
    "liquidity": [{"category": "highly", "count": 500, "totalValue": 1.0e12, "totalWeight": 99.9}],
    "fairValue": [{"level": 1, "count": 500, "totalValue": 1.0e12, "totalWeight": 99.9}],
    "defaults": {"defaults": 0, "arrears": 0, "total": 0},
    "serviceProviders": [{"providerRole": "Investment Adviser", "providerName": "Vanguard", "isAffiliated": True}],
}
FEE_ANALYSIS = {
    "feeRecords": [
        {"reportDate": "2025-12-31", "isExpenseLimitation": False, "isExpenseReducedOrWaived": False,
         "isFeesWaivedRecoupable": False, "monthlyAvgNetAssets": 1.4e12, "aggregateCommission": 1.0e6,
         "creationUnitShares": 25000.0, "avgPctPurchasedInKind": 80.0, "avgPctRedeemedInKind": 90.0,
         "hasLineOfCredit": True, "lineOfCreditSize": 4.0e9},
    ],
    "brokerCommissions": [{"providerName": "A Broker", "commission": 1.0e5, "isAffiliated": False}],
}
SERVICE_PROVIDERS = {
    "providersByYear": [
        {"reportDate": "2025-12-31", "providers": [
            {"providerRole": "Investment Adviser", "providerName": "The Vanguard Group, Inc.", "isAffiliated": True,
             "fileNumber": "801-11953", "crdNumber": "105958", "lei": None, "commission": None},
            {"providerRole": "Custodian", "providerName": "A Bank", "isAffiliated": False,
             "fileNumber": None, "crdNumber": None, "lei": None, "commission": None},
        ]},
    ],
}
BY_ADVISER = {
    "page": 1, "pageSize": 500, "totalPages": 1, "totalResults": 1,
    "advisers": [{"name": "The Vanguard Group, Inc.", "crdNumber": "105958", "lei": None}],
    "data": [{"ticker": "VOO", "name": "500 Index Fund", "fundKind": "mf", "cik": TRUST, "seriesId": 2839,
              "crdNumber": "105958"}],
}


def _client(vendor, clock):

    return vendor.client()


def _refresh(tmp_path, vendor, clock, endpoints=FUND_ENDPOINTS):
    from quantlab.acquisition.stockfit.refresh import refresh_raw_tier

    return refresh_raw_tier(tmp_path, client=_client(vendor, clock), endpoints=endpoints)


# -- seam 1: the raw tier ---------------------------------------------------------


def test_two_series_of_one_trust_are_two_fund_series_each_pulled_by_its_own_ticker(env, tmp_path, clock):
    from quantlab.dataset.stockfit import raw

    vendor = _vendor(clock)
    result = _refresh(tmp_path, vendor, clock)

    root = tmp_path / "stockfit"
    assert not result.failed, result.summary()
    assert raw.fund_series(root) == [VOO, OTHER]
    assert raw.entities(root, "fund_profile") == [VOO, OTHER]
    assert raw.read_pull(root, "fund_profile", VOO) == fixture("fund_profile_voo")
    assert raw.read_pull(root, "fund_profile", OTHER)[0]["seriesId"] == 2840
    assert raw.read_pull(root, "fund_flows", VOO) == fixture("fund_flows_voo")
    assert raw.read_pull(root, "fund_fees", VOO) == fixture("fund_fees_voo")["data"]
    assert raw.read_pull(root, "fund_performance", VOO) == fixture("fund_performance_voo")["data"]
    assert [params.get("symbol") for params in vendor.requested("/api/fund/flows")] == ["VOO", "VXX0"]
    assert all("cik" not in params for params in vendor.requested("/api/fund/profile"))


def test_the_n_cen_paths_land_raw_and_the_adviser_lookup_follows_the_service_providers(env, tmp_path, clock):
    from quantlab.dataset.stockfit import raw

    _refresh(tmp_path, _vendor(clock), clock)

    root = tmp_path / "stockfit"
    assert raw.read_pull(root, "fund_health", VOO) == FUND_HEALTH
    assert raw.read_pull(root, "fund_fee_analysis", VOO) == FEE_ANALYSIS
    assert raw.read_pull(root, "fund_service_providers", VOO) == SERVICE_PROVIDERS
    assert raw.entities(root, "fund_by_adviser") == ["105958"]
    assert raw.read_pull(root, "fund_by_adviser", "105958") == BY_ADVISER["data"]


def test_n_port_holdings_land_once_per_series_and_quarter(env, tmp_path, clock):
    from quantlab.dataset.stockfit import raw

    vendor = _vendor(clock)
    _refresh(tmp_path, vendor, clock)
    asked = len(vendor.requested("/api/fund/holdings"))
    clock.advance(days=1)
    _refresh(tmp_path, vendor, clock)

    root = tmp_path / "stockfit"
    assert asked == 28 + 2  # every quarter VOO's profile lists, and the two of the second series
    assert len(vendor.requested("/api/fund/holdings")) == asked  # a quarter asked once is not asked again
    assert raw.entities(root, "fund_holdings") == [f"{VOO}_2026-06-30"]
    assert raw.read_pull(root, "fund_holdings", f"{VOO}_2026-06-30")[0]["cusip"] == "67066G104"


def test_daily_etf_holdings_accumulate_as_raw_snapshots(env, tmp_path, clock):
    from quantlab.dataset.stockfit import raw

    vendor = _vendor(clock)
    _refresh(tmp_path, vendor, clock)
    first = clock.now()
    clock.advance(days=1)
    _refresh(tmp_path, vendor, clock)  # the issuer published nothing new
    next_day = _one_page(fixture("fund_holdings_daily_voo_page1"))
    next_day.update(reportDate="2026-10-01")  # SYNTHETIC
    next_day["data"][0]["balance"] += 1000
    vendor.bodies[_key("/api/fund/holdings/daily", symbol="VOO")] = next_day
    clock.advance(days=1)
    _refresh(tmp_path, vendor, clock)

    root = tmp_path / "stockfit"
    assert raw.entities(root, "fund_holdings_daily") == [VOO]  # only the series with a supported ticker
    assert raw.pulls(root, "fund_holdings_daily", VOO) == [first, clock.now()]
    assert raw.read_pull_as_of(root, "fund_holdings_daily", VOO, first)[0]["balance"] == 635561927
    assert raw.read_pull(root, "fund_holdings_daily", VOO)[0]["balance"] == 635561927 + 1000


def test_a_trusts_new_n_port_re_pulls_the_n_port_endpoints_of_each_of_its_series(env, tmp_path, clock):
    from tests.stockfit_fixtures import filing_row

    vendor = _vendor(clock)
    _refresh(tmp_path, vendor, clock, endpoints=(*FUND_ENDPOINTS, "filings_latest"))
    clock.advance(days=1)
    vendor.calls.clear()
    _refresh(tmp_path, vendor, clock, endpoints=(*FUND_ENDPOINTS, "filings_latest"))
    assert vendor.requested("/api/fund/profile") == []  # nothing filed: not due for 30 days

    clock.advance(days=1)
    vendor.lists["/api/filings/latest"] = [  # the feed names the trust, not the series
        filing_row("NPORT-P", TRUST, clock.now().date().isoformat(), clock.now().isoformat()),
    ]
    clock.advance(hours=1)
    vendor.calls.clear()
    _refresh(tmp_path, vendor, clock, endpoints=(*FUND_ENDPOINTS, "filings_latest"))

    assert sorted(params["symbol"] for params in vendor.requested("/api/fund/profile")) == ["VOO", "VXX0"]
    assert len(vendor.requested("/api/fund/flows")) == 2
    assert vendor.requested("/api/fund/fees") == []  # the feed does not follow N-1A


# -- seam 2: the fund panels --------------------------------------------------------


def _panel(tmp_path, cls_name, start="2026-01-01", end="2026-12-31", **config):
    from quantlab.dataset import config as configs
    from quantlab.dataset.stockfit import funds

    cls = getattr(funds, cls_name)
    dataset = cls(configs.StockFitFundConfig(
        zarr_file_path=str(tmp_path / f"{cls_name}.zarr"), raw_data_dir_path=str(tmp_path / "stockfit"),
        start_date=start, end_date=end, **config,
    ))
    dataset.update()
    return dataset.panel(start, end).load()


def _at(panel, variable, day, symbol=VOO):
    return panel[variable].sel(timestamp=pd.Timestamp(day), symbol=symbol).item()


def test_two_series_of_one_trust_are_two_symbols_of_the_n_port_summary(env, tmp_path, clock):
    _refresh(tmp_path, _vendor(clock), clock)
    panel = _panel(tmp_path, "StockFitFundProfileDataset")

    assert panel["symbol"].values.tolist() == [VOO, OTHER]
    assert _at(panel, "net_assets", "2026-09-01") == 1671231561960
    assert _at(panel, "net_assets", "2026-09-01", OTHER) == 167123156196.0


def test_n_port_data_appears_60_days_after_the_quarter_end(env, tmp_path, clock):
    _refresh(tmp_path, _vendor(clock), clock)
    panel = _panel(tmp_path, "StockFitFundProfileDataset")

    # 2026-06-30 + 60 days is Saturday 2026-08-29: shown from Monday 2026-08-31.
    assert _at(panel, "net_assets", "2026-08-28") == 1421263311403  # the March quarter, out since 2026-06-01
    assert _at(panel, "net_assets", "2026-08-31") == 1671231561960
    assert _at(panel, "holding_count", "2026-08-31") == 508
    assert pd.Timestamp(_at(panel, "report_date", "2026-08-31")) == pd.Timestamp("2026-06-30")
    assert panel["timestamp"].values.max() == np.datetime64("2026-11-13")  # the day before the pull, on NYSE's days


def test_monthly_flows_appear_60_days_after_the_quarter_end(env, tmp_path, clock):
    _refresh(tmp_path, _vendor(clock), clock)
    panel = _panel(tmp_path, "StockFitFundFlowsDataset")

    assert panel["symbol"].values.tolist() == [VOO]  # the second series serves no flows
    assert _at(panel, "flow_sales_mon1", "2026-08-31") == 40632647042.0601
    assert _at(panel, "flow_sales_mon1", "2026-08-28") == 49491099928.3702


def test_fees_show_from_their_filing_date(env, tmp_path, clock):
    _refresh(tmp_path, _vendor(clock), clock)
    panel = _panel(tmp_path, "StockFitFundFeesDataset")

    assert pd.Timestamp(_at(panel, "filed_date", "2026-04-27")) == pd.Timestamp("2025-04-28")
    assert pd.Timestamp(_at(panel, "filed_date", "2026-04-28")) == pd.Timestamp("2026-04-28")
    assert _at(panel, "total_expense_ratio", "2026-04-28") == 0.0003
    assert np.isnan(_at(panel, "fee_waiver", "2026-04-28"))


def test_n_cen_fields_appear_75_days_after_the_reporting_period(env, tmp_path, clock):
    _refresh(tmp_path, _vendor(clock), clock)
    panel = _panel(tmp_path, "StockFitFundNcenDataset")

    # 2025-12-31 + 75 days is Monday 2026-03-16.
    assert np.isnan(_at(panel, "is_index_fund", "2026-03-13"))
    assert _at(panel, "is_index_fund", "2026-03-16") == 1.0
    assert _at(panel, "did_lend", "2026-03-16") == 0.0
    assert _at(panel, "monthly_avg_net_assets", "2026-03-16") == 1.4e12


def test_a_fund_panel_keeps_only_the_configured_series(env, tmp_path, clock):
    _refresh(tmp_path, _vendor(clock), clock)
    panel = _panel(tmp_path, "StockFitFundProfileDataset", symbols=(OTHER,))

    assert panel["symbol"].values.tolist() == [OTHER]
