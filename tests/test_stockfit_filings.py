"""StockFit's filing list in the raw tier, and the filing panel built from it.

Seam 1, the raw-tier refresh entry: ``tests.stockfit_fixtures.FakeVendor``
plays StockFit and serves each company's ``/api/filings`` list. Seam 2: the
filing panel is built from that raw tier on disk. The recorded GOOGL rows are
VERBATIM; rows made up for a placement rule are marked SYNTHETIC.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

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


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("STOCKFIT_API_KEY", KEY)


def filing(form, accepted, filed, *, amendment=False, events=None, accession="0000000000-26-000001"):
    """One ``/api/filings`` row in the vendor's shape; SYNTHETIC values."""
    row = {
        "type": form,
        "accessionNumber": accession,
        "url": "https://www.sec.gov/Archives/edgar/data/1/x.htm",
        "dateFiled": filed,
        "acceptedAt": accepted,
        "xbrl": False,
        "amendment": amendment,
    }
    if events is not None:
        row["events"] = events
    return row


def _vendor(clock, filings_by_cik):
    vendor = FakeVendor(clock, companies=[company_row(cik, f"Company {cik}") for cik in filings_by_cik])
    for cik, rows in filings_by_cik.items():
        vendor.bodies[("/api/filings", frozenset({("cik", str(cik))}))] = {
            "page": 1, "pageSize": 100, "totalPages": 1, "totalResults": len(rows), "data": rows,
        }
    return vendor


def _refresh(download_dir, vendor, clock, ciks=None):
    from quantlab.acquisition.stockfit.refresh import refresh_raw_tier

    client = vendor.client()
    return refresh_raw_tier(download_dir, client=client, endpoints=("symbols", "filings"), ciks=ciks)


def _dataset(download_dir, tmp_path, start="2026-01-01", end=None):
    from quantlab.dataset.config import StockFitFilingsConfig
    from quantlab.dataset.stockfit.filings import StockFitFilingsDataset

    config = StockFitFilingsConfig(
        zarr_file_path=str(tmp_path / "stockfit_filings_1d.zarr"),
        raw_data_dir_path=str(download_dir / "stockfit"),
        start_date=start,
        end_date=end,
    )
    return StockFitFilingsDataset(config)


def _panel(download_dir, start, end):
    _dataset(download_dir, download_dir).update()
    return _dataset(download_dir, download_dir).panel(start, end).load()


def test_the_refresh_keeps_each_companys_filing_metadata_in_the_raw_tier(env, tmp_path):
    from quantlab.dataset.stockfit import raw
    from quantlab.dataset.stockfit.filings import read_filings

    clock = FakeClock()
    recorded = fixture("filings_googl_page1")["data"]
    result = _refresh(tmp_path, _vendor(clock, {GOOGL: recorded}), clock)

    assert not result.failed, result.summary()
    root = tmp_path / "stockfit"
    assert raw.read_pull(root, "filings", str(GOOGL)) == recorded
    rows = read_filings(root)
    by_accession = {row["accession_number"]: row for row in rows.iter_rows(named=True)}
    ten_q = by_accession["0001652044-26-000084"]  # VERBATIM: GOOGL's 2026-Q3 10-Q
    assert ten_q["cik"] == GOOGL and ten_q["type"] == "10-Q"
    assert ten_q["date_filed"] == date(2026, 10, 29)
    assert ten_q["accepted_at"] == datetime(2026, 10, 29, 0, 29, 6, tzinfo=UTC)
    assert ten_q["period_end_date"] == date(2026, 9, 30)
    assert rows.height == len(recorded)


def test_a_weekend_filing_lands_on_the_next_nyse_trading_day(env, tmp_path):
    clock = FakeClock()
    saturday = filing("8-K", None, "2026-11-07", events=["earnings"])  # SYNTHETIC: no acceptance time
    _refresh(tmp_path, _vendor(clock, {PLUG: [saturday]}), clock)

    panel = _panel(tmp_path, "2026-11-05", "2026-11-12")

    days = [str(day)[:10] for day in panel["timestamp"].values]
    assert days == ["2026-11-05", "2026-11-06", "2026-11-09", "2026-11-10", "2026-11-11", "2026-11-12"]
    assert panel["form_8_k"].sel(symbol=PLUG).values.tolist() == [0, 0, 1, 0, 0, 0]
    assert panel["event_earnings"].sel(symbol=PLUG).values.tolist() == [False, False, True, False, False, False]


def _days_with(panel, variable, symbol):
    series = panel[variable].sel(symbol=symbol)
    return {str(day)[:10]: value for day, value in zip(series["timestamp"].values, series.values.tolist()) if value}


def test_a_filing_accepted_at_or_after_the_close_lands_on_the_next_trading_day(env, tmp_path):
    clock = FakeClock(datetime(2026, 12, 2, 14, tzinfo=UTC))
    recorded = fixture("filings_googl_page1")["data"]
    after_close = next(row for row in recorded if row["accessionNumber"] == "0001193125-26-449835")
    assert after_close["acceptedAt"] == "2026-11-05T21:10:44Z"  # VERBATIM: 16:10 ET on Thursday
    rows = [
        after_close,
        filing("10-Q", "2026-11-05T20:59:59Z", "2026-11-05", accession="a-1"),  # SYNTHETIC: 15:59:59 ET
        filing("10-Q", "2026-11-10T21:00:00Z", "2026-11-10", accession="a-2"),  # SYNTHETIC: 16:00 ET sharp
        filing("4", "2026-11-27T18:30:00Z", "2026-11-27", accession="a-3"),  # SYNTHETIC: 13:30 ET, a 13:00 close
        filing("4", "2026-11-27T17:59:00Z", "2026-11-27", accession="a-4"),  # SYNTHETIC: 12:59 ET, before it
    ]
    _refresh(tmp_path, _vendor(clock, {GOOGL: rows}), clock)

    panel = _panel(tmp_path, "2026-11-05", "2026-11-30")

    assert _days_with(panel, "form_8_k", GOOGL) == {"2026-11-06": 1}
    assert _days_with(panel, "event_earnings", GOOGL) == {}
    assert _days_with(panel, "event_other", GOOGL) == {"2026-11-06": True}
    assert _days_with(panel, "form_10_q", GOOGL) == {"2026-11-05": 1, "2026-11-11": 1}
    assert _days_with(panel, "form_4", GOOGL) == {"2026-11-27": 1, "2026-11-30": 1}


def test_a_day_counts_each_form_and_flags_an_amendment(env, tmp_path):
    clock = FakeClock()
    rows = [  # SYNTHETIC, all accepted on Tuesday 2026-11-10 before the close
        filing("10-K", "2026-11-10T15:00:00Z", "2026-11-10", accession="b-1"),
        filing("10-K", "2026-11-10T16:00:00Z", "2026-11-10", amendment=True, accession="b-2"),
        filing("144", "2026-11-10T17:00:00Z", "2026-11-10", accession="b-3"),
        filing("DEF 14A", "2026-11-12T15:00:00Z", "2026-11-12", accession="b-4"),
    ]
    _refresh(tmp_path, _vendor(clock, {PLUG: rows}), clock)

    panel = _panel(tmp_path, "2026-11-09", "2026-11-13")

    assert _days_with(panel, "form_10_k", PLUG) == {"2026-11-10": 2}
    assert _days_with(panel, "form_other", PLUG) == {"2026-11-10": 1}
    assert _days_with(panel, "form_def_14a", PLUG) == {"2026-11-12": 1}
    assert _days_with(panel, "amendment", PLUG) == {"2026-11-10": True}
    assert panel["form_10_q"].sel(symbol=PLUG).values.tolist() == [0, 0, 0, 0, 0]


def test_the_panel_ends_on_the_day_before_the_latest_pull(env, tmp_path):
    clock = FakeClock(datetime(2026, 11, 12, 23, tzinfo=UTC))  # Thursday, after the close
    rows = [  # SYNTHETIC
        filing("8-K", "2026-11-11T15:00:00Z", "2026-11-11", accession="c-1"),
        filing("8-K", "2026-11-12T15:00:00Z", "2026-11-12", accession="c-2"),
    ]
    _refresh(tmp_path, _vendor(clock, {PLUG: rows}), clock)

    panel = _panel(tmp_path, "2026-11-09", "2026-11-30")

    assert str(panel["timestamp"].values[-1])[:10] == "2026-11-11"
    assert _days_with(panel, "form_8_k", PLUG) == {"2026-11-11": 1}


def test_a_company_not_pulled_yet_is_missing_not_zero_until_its_first_pull(env, tmp_path):
    import math

    clock = FakeClock()
    rows = {cik: [filing("8-K", "2026-11-10T15:00:00Z", "2026-11-10", accession=f"d-{cik}")] for cik in (PLUG, GOOGL)}
    vendor = _vendor(clock, rows)  # SYNTHETIC filings
    _refresh(tmp_path, vendor, clock, ciks=(PLUG,))  # a capped backfill: GOOGL waits for the next run

    dataset = _dataset(tmp_path, tmp_path, start="2026-11-02")
    dataset.update()
    panel = dataset.panel("2026-11-02", "2026-11-13").load()
    assert GOOGL not in panel["symbol"].values.tolist()
    shown = panel["form_8_k"].reindex(symbol=[PLUG, GOOGL])  # read against a universe: missing, never 0
    assert math.isnan(shown.sel(symbol=GOOGL, timestamp="2026-11-10").item())

    clock.advance(days=1)
    _refresh(tmp_path, vendor, clock)
    _dataset(tmp_path, tmp_path, start="2026-11-02").update()
    panel = _dataset(tmp_path, tmp_path, start="2026-11-02").panel("2026-11-02", "2026-11-13").load()
    assert _days_with(panel, "form_8_k", GOOGL) == {"2026-11-10": 1}
    assert _days_with(panel, "form_8_k", PLUG) == {"2026-11-10": 1}
