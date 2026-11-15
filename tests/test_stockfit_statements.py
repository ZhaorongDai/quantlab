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
    from quantlab.acquisition.stockfit.client import StockFitClient
    from quantlab.acquisition.stockfit.refresh import refresh_raw_tier

    clock = vendor.clock if vendor is not None else FakeClock()
    vendor = vendor if vendor is not None else FakeVendor(clock)
    client = StockFitClient(http_get=vendor, sleep=clock.sleep, now=clock.now)
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
                 "assets", "stockholdersEquity", "endCash", "impairments"):
        assert _on(panel, "2026-11-13", name) == pytest.approx(vendor_ttm[name]), name
    assert _on(panel, "2026-11-13", "period_end") == pd.Timestamp("2026-09-30").value


def test_a_ttm_value_changes_when_a_quarter_in_it_is_restated(env, tmp_path):
    panel = _panel(tmp_path, _raw_tier(tmp_path, _restated(FakeVendor(FakeClock()))), period="ttm")

    before = _on(panel, "2026-11-11", "revenue")
    assert _on(panel, "2026-11-12", "revenue") - before == 210_000_000 - 200_602_000


def test_annual_statements_stay_in_the_raw_tier(tmp_path):
    with pytest.raises(ValueError, match="Annual statements stay in the raw tier"):
        _panel(tmp_path, tmp_path, period="annual")
