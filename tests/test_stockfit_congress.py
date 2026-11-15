"""StockFit's Congress trades: the ``since`` feeds in the raw tier, and the panel on the CIK axis.

Seam 1, the raw-tier refresh entry, with ``tests.stockfit_fixtures.FakeVendor``
serving the House and Senate feeds; seam 2, the panel built from that raw
tier on disk. Feed rows are VERBATIM field names with SYNTHETIC values.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.stockfit_fixtures import KEY, FakeClock, FakeVendor, company_row, fixture

HOUSE = "/api/congress/house/trades/recent"
SENATE = "/api/congress/senate/trades/recent"

#: SYNTHETIC companies: META held the ticker FB until a change after 2022-03-31;
#: a later company holds FB today.
META, LATER_FB, ZTS = 1326801, 999001, 1555280


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("STOCKFIT_API_KEY", KEY)


@pytest.fixture
def clock():
    return FakeClock()


def _client(vendor, clock):
    from quantlab.acquisition.stockfit.client import StockFitClient

    return StockFitClient(http_get=vendor, sleep=clock.sleep, now=clock.now)


def _refresh_congress(tmp_path, vendor, clock):
    from quantlab.acquisition.stockfit.refresh import refresh_raw_tier

    return refresh_raw_tier(
        tmp_path, client=_client(vendor, clock),
        endpoints=("symbols", "company_delisted", "congress_house_trades", "congress_senate_trades"),
    )


def _trade(ticker, kind, day, low, high, *, confidence="high", owner="self"):
    return {  # VERBATIM field names, SYNTHETIC values
        "owner": owner, "assetName": f"{ticker} Common Stock", "assetTypeCode": "ST", "ticker": ticker,
        "transactionType": kind, "transactionDate": day, "notificationDate": None,
        "amountRange": f"${low:,} - ${high:,}" if high else f"Over ${low:,}",
        "amountMin": low, "amountMax": high, "capGainsOver200": None, "description": None,
        "resolvedTicker": ticker if confidence != "none" else None, "linkConfidence": confidence,
    }


def _filing(filing_id, filed, member, trades, *, status="parsed", amendment=False):
    return {
        "filingId": filing_id, "filingDate": filed, "filingUrl": f"https://example.org/{filing_id}.pdf",
        "parseStatus": status, "isAmendment": amendment,
        "member": {"bioguideId": member, "name": member, "party": "Independent", "state": "VT", "district": None},
        "transactionCount": len(trades), "transactions": trades,
    }


def _companies():
    meta = company_row(META, "Meta Platforms, Inc.")
    meta.update(symbols=["META"], priorTickers=[{"symbol": "FB", "exchange": "Nasdaq", "lastSeen": "2022-03-31"}])
    later = company_row(LATER_FB, "Later FB Holder Inc")
    later.update(symbols=["FB"])
    zoetis = company_row(ZTS, "Zoetis Inc.")
    zoetis.update(symbols=["ZTS"])
    return [meta, later, zoetis]


def _vendor(clock, house, senate=()):
    vendor = FakeVendor(clock, companies=_companies())
    vendor.lists[HOUSE] = list(house)
    vendor.lists[SENATE] = list(senate)
    return vendor


def test_the_feeds_are_pulled_from_the_archive_start_then_from_the_watermark(env, tmp_path, clock):
    from quantlab.dataset.stockfit import raw

    vendor = _vendor(clock, fixture("congress_house_trades_recent")["data"],
                     fixture("congress_senate_trades_recent")["data"])
    result = _refresh_congress(tmp_path, vendor, clock)
    clock.advance(days=3)
    _refresh_congress(tmp_path, vendor, clock)

    root = tmp_path / "stockfit"
    assert not result.failed, result.summary()
    first, second = [params["since"] for params in vendor.requested(HOUSE) if str(params["page"]) == "1"]
    assert first == "2012-01-01"
    assert second == "2026-10-17"  # the last pull's day, 30 days back
    assert raw.read_pull(root, "congress_house_trades", "all") == fixture("congress_house_trades_recent")["data"]
    assert len(raw.pulls(root, "congress_senate_trades", "all")) == 1  # the re-pull was unchanged


def test_paper_filings_without_transactions_are_kept_raw(env, tmp_path, clock):
    from quantlab.dataset.stockfit import raw

    paper = _filing("P1", "2024-03-04", "M1", [], status="unparsed_paper")
    _refresh_congress(tmp_path, _vendor(clock, [paper]), clock)

    assert raw.read_pull(tmp_path / "stockfit", "congress_house_trades", "all") == [paper]


def _panel(tmp_path, clock, house, senate=()):
    from quantlab.dataset.config import StockFitCongressTradesConfig
    from quantlab.dataset.stockfit.congress import StockFitCongressTradesDataset

    result = _refresh_congress(tmp_path, _vendor(clock, house, senate), clock)
    assert not result.failed, result.summary()
    config = StockFitCongressTradesConfig(
        zarr_file_path=str(tmp_path / "stockfit_congress_1d.zarr"),
        raw_data_dir_path=str(tmp_path / "stockfit"),
        start_date="2021-01-04",
    )
    StockFitCongressTradesDataset(config).update()
    return StockFitCongressTradesDataset(config).panel("2021-01-04", "2026-11-13").load()


def _at(panel, name, day, cik):
    return float(panel[name].sel(timestamp=day, symbol=cik).item())


def test_a_trade_under_a_reassigned_ticker_is_placed_on_its_trade_date_holder(env, tmp_path, clock):
    panel = _panel(tmp_path, clock, [
        _filing("H1", "2021-06-08", "M1", [_trade("FB", "purchase", "2021-06-01", 1001, 15000)]),
        _filing("H2", "2023-06-06", "M1", [_trade("FB", "sale", "2023-06-01", 15001, 50000)]),
    ])

    assert _at(panel, "purchase_min", "2021-06-08", META) == 1001
    assert _at(panel, "sale_max", "2023-06-06", LATER_FB) == 50000
    assert _at(panel, "sale_max", "2023-06-06", META) == 0
    assert _at(panel, "purchase_min", "2021-06-08", LATER_FB) == 0


def test_low_confidence_and_unresolved_trades_never_reach_the_panel(env, tmp_path, clock):
    panel = _panel(tmp_path, clock, [
        _filing("H1", "2023-06-06", "M1", [
            _trade("META", "purchase", "2023-06-01", 1001, 15000, confidence="medium"),
            _trade("META", "purchase", "2023-06-01", 1001, 15000, confidence="low"),
            _trade("META", "purchase", "2023-06-01", 1001, 15000, confidence="none"),
            _trade("ZZZZ", "purchase", "2023-06-01", 1001, 15000),
            # FB weeks after its old holder was last seen: either company may have held it.
            _trade("FB", "purchase", "2022-05-02", 1001, 15000),
        ]),
        _filing("H2", "2023-06-07", "M1", [_trade("ZTS", "purchase", "2023-06-01", 1001, 15000)]),
    ])

    assert panel.sizes["symbol"] == 1 and int(panel["symbol"].item()) == ZTS
    assert float(panel["purchases"].sum()) == 1


def test_a_days_trades_sum_their_amount_ranges_per_company_on_the_filing_day(env, tmp_path, clock):
    panel = _panel(tmp_path, clock, [
        _filing("H1", "2023-06-06", "M1", [
            _trade("ZTS", "purchase", "2023-06-01", 1001, 15000),
            _trade("ZTS", "purchase", "2023-06-01", 1001, 15000),  # a second, identical trade counts
            _trade("ZTS", "sale_partial", "2023-06-01", 50000001, None),  # an open band counts its floor
            _trade("ZTS", "exchange", "2023-06-01", 1001, 15000),
        ]),
        _filing("H2", "2023-06-10", "M2", [_trade("ZTS", "sale", "2023-06-02", 15001, 50000)]),  # a Saturday
    ], senate=[
        _filing("S1", "2023-06-06", "S9", [_trade("ZTS", "purchase", "2023-05-30", 15001, 50000, owner="spouse")]),
    ])

    day = "2023-06-06"
    assert _at(panel, "purchase_min", day, ZTS) == 1001 + 1001 + 15001
    assert _at(panel, "purchase_max", day, ZTS) == 15000 + 15000 + 50000
    assert _at(panel, "purchases", day, ZTS) == 3
    assert _at(panel, "members_buying", day, ZTS) == 2
    assert _at(panel, "sale_min", day, ZTS) == 50000001
    assert _at(panel, "sale_max", day, ZTS) == 50000001
    assert _at(panel, "sales", "2023-06-12", ZTS) == 1  # the Saturday filing, on Monday
    assert _at(panel, "purchases", "2023-06-07", ZTS) == 0


def test_a_trade_an_amendment_repeats_is_counted_on_its_first_filing_only(env, tmp_path, clock):
    trade = _trade("ZTS", "purchase", "2023-06-01", 1001, 15000)
    panel = _panel(tmp_path, clock, [
        _filing("H1", "2023-06-06", "M1", [trade]),
        _filing("H2", "2023-06-20", "M1", [trade, _trade("ZTS", "sale", "2023-06-15", 1001, 15000)], amendment=True),
    ])

    assert _at(panel, "purchases", "2023-06-06", ZTS) == 1
    assert _at(panel, "purchases", "2023-06-20", ZTS) == 0
    assert _at(panel, "sales", "2023-06-20", ZTS) == 1
    assert np.isfinite(panel["purchase_max"].values).all()
