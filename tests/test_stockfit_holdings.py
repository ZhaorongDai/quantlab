"""StockFit 13F: the raw-tier pulls and the per-security quarterly panel on the CUSIP axis.

Seam 1, the raw-tier refresh entry, with HTTP replaced by
``tests.stockfit_fixtures.FakeVendor``; seam 2, the panel built from the raw
tier that refresh wrote. Holder rows are SYNTHETIC unless they come from a
recorded fixture; field names are VERBATIM StockFit.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import pytest

from tests.stockfit_fixtures import KEY, FakeClock, FakeVendor, company_row

HOLDERS = "/api/ownership/institutional-holders"
FILINGS = "/api/filings"
PORTFOLIO = "/api/ownership/portfolio"

#: SYNTHETIC companies and managers.
ALPHA, BETA = 1652044, 320193
MANAGER_A, MANAGER_B = 9001, 9002
#: Alphabet's class A and class C CUSIPs (VERBATIM), and a CUSIP of digits only.
CLASS_A, CLASS_C, DIGITS = "02079K305", "02079K107", "037833100"


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("STOCKFIT_API_KEY", KEY)


@pytest.fixture
def clock():
    return FakeClock()


def holder(manager: int, cusip: str, quarter: str, accession: str, shares: float, **fields) -> dict:
    """Return one row of ``/api/ownership/institutional-holders`` in the vendor's shape."""
    return {  # SYNTHETIC values, VERBATIM field names
        "managerName": f"Manager {manager}",
        "managerCik": manager,
        "value": shares * 10.0,
        "shares": shares,
        "shareType": "SH",
        "investmentDiscretion": "SOLE",
        "soleVoting": shares,
        "sharedVoting": 0,
        "noneVoting": 0,
        "reportDate": quarter,
        "prevShares": None,
        "prevValue": None,
        "pctOfShares": 1.0,
        "accessionNumber": accession,
        "putCall": None,
        "cusip": cusip,
        "symbol": "SYM",
        **fields,
    }


def filing(accession: str, period: str, filed: str, accepted: str | None, amendment: bool = False) -> dict:
    """Return one row of ``/api/filings?type=13F-HR`` in the vendor's shape."""
    return {  # SYNTHETIC values, VERBATIM field names
        "type": "13F-HR",
        "accessionNumber": accession,
        "url": "https://www.sec.gov/Archives/edgar/data/0/0/primary_doc.xml",
        "dateFiled": filed,
        "acceptedAt": accepted,
        "xbrl": False,
        "amendment": amendment,
        "periodEndDate": period,
    }


def paged(rows: list[dict], **top) -> dict:
    return {"page": 1, "pageSize": 500, "totalPages": 1 if rows else 0, "totalResults": len(rows), **top, "data": rows}


def key(path: str, **params) -> tuple[str, frozenset]:
    return path, frozenset((k, str(v)) for k, v in params.items())


def serve_holders(vendor: FakeVendor, cik: int, quarter: str, rows: list[dict]) -> None:
    vendor.bodies[key(HOLDERS, cik=cik, reportDate=quarter)] = paged(rows)


def serve_filings(vendor: FakeVendor, manager: int, rows: list[dict]) -> None:
    vendor.bodies[key(FILINGS, cik=manager, type="13F-HR")] = paged(rows)


def serve_portfolio(vendor: FakeVendor, manager: int, quarter: str, rows: list[dict]) -> None:
    vendor.bodies[key(PORTFOLIO, cik=manager, reportDate=quarter)] = paged(
        rows, managerName=f"Manager {manager}", managerCik=manager, reportDate=quarter
    )


def _vendor(clock) -> FakeVendor:
    return FakeVendor(clock, companies=[company_row(ALPHA, "Alphabet Inc."), company_row(BETA, "Beta Corp")])


THIRTEEN_F = ("symbols", "institutional_holders", "manager_13f_filings", "manager_portfolio")


def _refresh(tmp_path, vendor, clock, *, end=date(2024, 7, 1), first_quarter=date(2024, 3, 31), names=THIRTEEN_F, runs=3):
    """Run the refresh ``runs`` times: holders land, then their managers' filings, then the portfolios."""
    from quantlab.acquisition.stockfit.endpoints import ENDPOINTS
    from quantlab.acquisition.stockfit.refresh import refresh_raw_tier

    rows = [
        replace(ENDPOINTS[name], first_quarter=first_quarter) if hasattr(ENDPOINTS[name], "first_quarter")
        else ENDPOINTS[name]
        for name in names
    ]
    client = vendor.client()
    for _ in range(runs):
        result = refresh_raw_tier(tmp_path, client=client, endpoints=rows, end=end)
        assert not result.failed, result.summary()
    return result


# -- seam 1: the raw tier ------------------------------------------------------


def test_holders_are_pulled_per_company_and_quarter_and_managers_from_their_rows(env, tmp_path, clock):
    from quantlab.dataset.stockfit import raw

    vendor = _vendor(clock)
    serve_holders(vendor, ALPHA, "2024-03-31", [
        holder(MANAGER_A, CLASS_A, "2024-03-31", "0000009001-24-000001", 100.0),
        holder(MANAGER_B, CLASS_C, "2024-03-31", "0000009002-24-000001", 50.0),
    ])
    serve_filings(vendor, MANAGER_A, [filing("0000009001-24-000001", "2024-03-31", "2024-05-10", None)])
    serve_portfolio(vendor, MANAGER_A, "2024-03-31", [{"cusip": CLASS_A, "shares": 100.0}])
    _refresh(tmp_path, vendor, clock, end=date(2024, 8, 20))

    root = tmp_path / "stockfit"
    # Only quarters whose filings are due 50 days before ``end``, newest first; each company and
    # quarter pulled once (2024-09-30 is not due yet).
    assert [(int(p["cik"]), p["reportDate"]) for p in vendor.requested(HOLDERS)] == [
        (BETA, "2024-06-30"), (ALPHA, "2024-06-30"), (BETA, "2024-03-31"), (ALPHA, "2024-03-31"),
    ]
    assert raw.entities(root, "institutional_holders") == [f"{ALPHA}_2024-03-31"]
    assert [row["cusip"] for row in raw.read_pull(root, "institutional_holders", f"{ALPHA}_2024-03-31")] == [
        CLASS_A, CLASS_C,
    ]
    assert raw.thirteen_f_managers(root) == [MANAGER_A, MANAGER_B]
    assert raw.entities(root, "manager_13f_filings") == [str(MANAGER_A)]
    # A manager's portfolios are pulled for the periods its 13F filings report.
    assert raw.entities(root, "manager_portfolio") == [f"{MANAGER_A}_2024-03-31"]
    assert raw.read_pull(root, "manager_portfolio", f"{MANAGER_A}_2024-03-31") == [{"cusip": CLASS_A, "shares": 100.0}]


def test_a_manager_stockfit_has_no_filing_list_for_takes_its_quarters_from_its_holder_rows(env, tmp_path, clock):
    from quantlab.dataset.stockfit import raw

    vendor = _vendor(clock)
    serve_holders(vendor, ALPHA, "2024-03-31", [holder(MANAGER_B, CLASS_C, "2024-03-31", "0000009002-24-000001", 50.0)])
    # /api/filings answers a manager outside StockFit's company lists with HTTP 400 "CIK '<cik>' not found"
    vendor.bodies.pop(key(FILINGS, cik=MANAGER_B, type="13F-HR"), None)
    serve_portfolio(vendor, MANAGER_B, "2024-03-31", [{"cusip": CLASS_C, "shares": 50.0}])
    _refresh(tmp_path, vendor, clock, end=date(2024, 8, 20))

    root = tmp_path / "stockfit"
    assert raw.entities(root, "manager_13f_filings") == []
    assert raw.read_pull(root, "manager_portfolio", f"{MANAGER_B}_2024-03-31") == [{"cusip": CLASS_C, "shares": 50.0}]


def test_a_managers_new_13f_filing_re_pulls_its_filings_and_the_holders_of_what_it_held(env, tmp_path, clock):
    from tests.stockfit_fixtures import filing_row

    vendor = _vendor(clock)
    serve_holders(vendor, ALPHA, "2024-03-31", [holder(MANAGER_A, CLASS_A, "2024-03-31", A1, 100.0)])
    serve_holders(vendor, BETA, "2024-03-31", [holder(MANAGER_B, DIGITS, "2024-03-31", B1, 5.0)])
    serve_filings(vendor, MANAGER_A, [filing(A1, "2024-03-31", "2024-05-10", None)])
    names = ("filings_latest", *THIRTEEN_F)
    _refresh(tmp_path, vendor, clock, names=names)
    before = len(vendor.calls)

    clock.advance(days=1)
    vendor.lists["/api/filings/latest"] = [filing_row("13F-HR", MANAGER_A, "2026-11-17", "2026-11-17T14:30:00Z")]
    _refresh(tmp_path, vendor, clock, names=names, runs=1)

    later = [(url.rsplit("/v1", 1)[1], params.get("cik")) for url, params, _ in vendor.calls[before:]]
    assert ("/api/ownership/institutional-holders", str(ALPHA)) in later
    assert ("/api/ownership/institutional-holders", str(BETA)) not in later
    assert ("/api/filings", str(MANAGER_A)) in later
    assert ("/api/filings", str(MANAGER_B)) not in later


# -- seam 2: the panel ---------------------------------------------------------


def _dataset(tmp_path, download_dir, **fields):
    from quantlab.dataset.config import StockFitHoldingsConfig
    from quantlab.dataset.stockfit.holdings import StockFitHoldingsDataset

    return StockFitHoldingsDataset(StockFitHoldingsConfig(
        zarr_file_path=str(tmp_path / "stockfit_holdings_1d.zarr"),
        raw_data_dir_path=str(download_dir / "stockfit"),
        **{"start_date": "2024-04-01", "end_date": "2024-12-31", **fields},
    ))


def _panel(tmp_path, download_dir, **fields):
    _dataset(tmp_path, download_dir, **fields).update()
    return _dataset(tmp_path, download_dir, **fields).panel("2024-05-01", "2024-08-30").load()


def _on(panel, variable, day, cusip) -> float:
    return float(panel[variable].sel(symbol=cusip, timestamp=day))


A1, B1 = "0000009001-24-000001", "0000009002-24-000001"


def test_two_share_classes_of_one_company_are_two_symbols_on_the_cusip_axis(env, tmp_path, clock):
    vendor = _vendor(clock)
    serve_holders(vendor, ALPHA, "2024-03-31", [
        holder(MANAGER_A, CLASS_A, "2024-03-31", A1, 100.0),
        holder(MANAGER_A, CLASS_A, "2024-03-31", A1, 9.0, putCall="Put"),  # an option, not a holding
        holder(MANAGER_B, CLASS_A, "2024-03-31", B1, 20.0, shareType="PRN"),  # a principal amount
        holder(MANAGER_B, CLASS_C, "2024-03-31", B1, 50.0),
    ])
    serve_holders(vendor, BETA, "2024-03-31", [holder(MANAGER_A, DIGITS, "2024-03-31", A1, 30.0)])
    serve_filings(vendor, MANAGER_A, [filing(A1, "2024-03-31", "2024-05-10", None)])
    serve_filings(vendor, MANAGER_B, [filing(B1, "2024-03-31", "2024-05-14", None)])
    _refresh(tmp_path / "dl", vendor, clock)

    panel = _panel(tmp_path, tmp_path / "dl")
    assert panel["symbol"].values.tolist() == [CLASS_C, CLASS_A, DIGITS]
    assert sorted(panel.data_vars) == ["holders", "quarter_end", "shares_held", "value_held"]
    assert _on(panel, "shares_held", "2024-05-15", CLASS_A) == 100.0
    assert _on(panel, "value_held", "2024-05-15", CLASS_A) == 1000.0
    assert _on(panel, "holders", "2024-05-15", CLASS_A) == 1.0
    assert _on(panel, "shares_held", "2024-05-15", CLASS_C) == 50.0
    assert _on(panel, "shares_held", "2024-05-15", DIGITS) == 30.0


def test_a_quarter_is_not_visible_before_quarter_end_plus_45_days(env, tmp_path, clock):
    vendor = _vendor(clock)
    serve_holders(vendor, ALPHA, "2024-03-31", [holder(MANAGER_A, CLASS_A, "2024-03-31", A1, 100.0)])
    serve_holders(vendor, ALPHA, "2024-06-30", [holder(MANAGER_A, CLASS_A, "2024-06-30", "a2", 120.0)])
    serve_filings(vendor, MANAGER_A, [
        filing(A1, "2024-03-31", "2024-04-02", None),  # filed early, still shown only from day 45
        filing("a2", "2024-06-30", "2024-07-15", "2024-07-15T14:00:00Z"),
    ])
    _refresh(tmp_path / "dl", vendor, clock, end=date(2024, 9, 1))

    panel = _panel(tmp_path, tmp_path / "dl")
    shares = panel["shares_held"].sel(symbol=CLASS_A)
    assert shares.sel(timestamp=slice(None, "2024-05-14")).isnull().all()
    assert float(shares.sel(timestamp="2024-05-15")) == 100.0  # 2024-03-31 + 45 days
    assert float(shares.sel(timestamp="2024-08-13")) == 100.0
    assert float(shares.sel(timestamp="2024-08-14")) == 120.0  # 2024-06-30 + 45 days
    assert str(panel["quarter_end"].sel(symbol=CLASS_A, timestamp="2024-08-14").values)[:10] == "2024-06-30"


def test_a_row_filed_after_day_45_counts_from_its_filings_available_date(env, tmp_path, clock):
    vendor = _vendor(clock)
    serve_holders(vendor, ALPHA, "2024-03-31", [
        holder(MANAGER_A, CLASS_A, "2024-03-31", A1, 100.0),
        holder(MANAGER_B, CLASS_A, "2024-03-31", "late-1", 40.0),  # accepted after the close
        holder(9003, CLASS_A, "2024-03-31", "late-2", 7.0),  # accepted a minute before the close
        holder(9004, CLASS_A, "2024-03-31", "half-day", 3.0),  # after a 13:00 close
        holder(9005, CLASS_A, "2024-03-31", "unknown", 1000.0),  # in no filing list nor feed: day 45
    ])
    serve_filings(vendor, MANAGER_A, [filing(A1, "2024-03-31", "2024-05-10", None)])
    serve_filings(vendor, MANAGER_B, [filing("late-1", "2024-03-31", "2024-05-20", "2024-05-20T20:05:00Z", True)])
    serve_filings(vendor, 9003, [filing("late-2", "2024-03-31", "2024-05-22", "2024-05-22T19:59:00Z")])
    serve_filings(vendor, 9004, [filing("half-day", "2024-03-31", "2024-07-03", "2024-07-03T17:30:00Z")])
    _refresh(tmp_path / "dl", vendor, clock)

    panel = _panel(tmp_path, tmp_path / "dl")
    shares = panel["shares_held"].sel(symbol=CLASS_A)
    assert float(shares.sel(timestamp="2024-05-15")) == 1100.0
    assert float(shares.sel(timestamp="2024-05-20")) == 1100.0
    assert float(shares.sel(timestamp="2024-05-21")) == 1140.0
    assert float(shares.sel(timestamp="2024-05-22")) == 1147.0
    assert float(shares.sel(timestamp="2024-07-03")) == 1147.0
    assert float(shares.sel(timestamp="2024-07-05")) == 1150.0
    assert float(panel["holders"].sel(symbol=CLASS_A, timestamp="2024-07-05")) == 5.0
    # The next quarter, without a row for the security, replaces it: nothing is shown.
    assert shares.sel(timestamp=slice("2024-08-14", None)).isnull().all()


def test_a_row_of_a_manager_without_a_filing_list_is_dated_by_the_recent_filings_feed(env, tmp_path, clock):
    vendor = _vendor(clock)
    serve_holders(vendor, ALPHA, "2024-03-31", [
        holder(MANAGER_A, CLASS_A, "2024-03-31", A1, 100.0),
        holder(MANAGER_B, CLASS_A, "2024-03-31", B1, 40.0),  # an amendment filed after day 45
    ])
    # /api/filings answers both managers with HTTP 400 "CIK not found": no filing list
    for manager in (MANAGER_A, MANAGER_B):
        vendor.bodies.pop(key(FILINGS, cik=manager, type="13F-HR"), None)
    feed = filing(B1, "2024-03-31", "2024-06-03", "2024-06-03T20:30:00Z", True)
    feed["url"] = f"https://www.sec.gov/Archives/edgar/data/{MANAGER_B}/000000900224000001/primary_doc.xml"
    vendor.lists["/api/filings/latest"] = [feed]
    _refresh(tmp_path / "dl", vendor, clock, names=("filings_latest", *THIRTEEN_F))

    panel = _panel(tmp_path, tmp_path / "dl")
    shares = panel["shares_held"].sel(symbol=CLASS_A)
    assert shares.sel(timestamp=slice(None, "2024-05-14")).isnull().all()
    assert float(shares.sel(timestamp="2024-05-15")) == 100.0  # no filing date known: day 45
    assert float(shares.sel(timestamp="2024-06-03")) == 100.0
    assert float(shares.sel(timestamp="2024-06-04")) == 140.0  # the feed's amendment, after the close


def test_manager_portfolios_are_raw_files_only(env, tmp_path, clock):
    import shutil

    vendor = _vendor(clock)
    serve_holders(vendor, ALPHA, "2024-03-31", [holder(MANAGER_A, CLASS_A, "2024-03-31", A1, 100.0)])
    serve_filings(vendor, MANAGER_A, [filing(A1, "2024-03-31", "2024-05-10", None)])
    serve_portfolio(vendor, MANAGER_A, "2024-03-31", [{"cusip": CLASS_A, "shares": 555.0}])
    _refresh(tmp_path / "dl", vendor, clock)
    with_portfolios = _panel(tmp_path / "a", tmp_path / "dl")

    shutil.rmtree(tmp_path / "dl" / "stockfit" / "manager_portfolio")
    assert _panel(tmp_path / "b", tmp_path / "dl").identical(with_portfolios)


def test_an_update_adds_a_new_security_with_its_history_as_a_rebuild_shows_it(env, tmp_path, clock):
    vendor = _vendor(clock)
    serve_holders(vendor, ALPHA, "2024-03-31", [holder(MANAGER_A, CLASS_A, "2024-03-31", A1, 100.0)])
    serve_filings(vendor, MANAGER_A, [filing(A1, "2024-03-31", "2024-05-10", None)])
    _refresh(tmp_path / "dl", vendor, clock)
    _dataset(tmp_path, tmp_path / "dl").update()

    serve_holders(vendor, BETA, "2024-06-30", [holder(MANAGER_B, DIGITS, "2024-06-30", "b2", 5.0)])
    serve_filings(vendor, MANAGER_B, [filing("b2", "2024-06-30", "2024-08-01", None)])
    clock.advance(days=1)
    _refresh(tmp_path / "dl", vendor, clock, end=date(2024, 9, 1))
    _dataset(tmp_path, tmp_path / "dl").update()

    updated = _dataset(tmp_path, tmp_path / "dl").panel("2024-05-01", "2024-08-30").load()
    assert _on(updated, "shares_held", "2024-08-14", DIGITS) == 5.0
    assert updated.identical(_panel(tmp_path / "rebuilt", tmp_path / "dl"))
