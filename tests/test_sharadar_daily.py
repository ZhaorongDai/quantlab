"""Sharadar DAILY valuations: a panel on the permaticker axis, in USD, refreshed by lastupdated.

A `FakeVendor` plays Sharadar; the real client, raw tier and dataset run
against it. DAILY's columns and the INDICATORS unit types are VERBATIM, every
row value is invented (`# SYNTHETIC`). TICKERS has no DAILY rows: DAILY covers
SF1's filers, so its tickers are mapped through the SF1 rows (VERBATIM labels
of the 2026-10-05 bulk TICKERS file: SEP, SFP, SF1, SF2, SF3B).
"""

from __future__ import annotations

import json

from datetime import date

import numpy as np
import pandas as pd
import pytest

from tests.sharadar_fixtures import (
    DAILY_COLUMNS,
    INDICATORS_COLUMNS,
    TICKERS_COLUMNS,
    FakeVendor,
    daily_row,
    tickers_row,
)

DAYS = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
PERMATICKERS = {"AAA": 101, "BBB": 202}  # SYNTHETIC

#: VERBATIM unit types of DAILY's indicators.
DAILY_UNITS = {
    "ev": "USD millions",
    "evebit": "ratio",
    "evebitda": "ratio",
    "marketcap": "USD millions",
    "pb": "ratio",
    "pe": "ratio",
    "ps": "ratio",
}


def _units(**overrides):
    units = {**DAILY_UNITS, **overrides}
    return [
        {
            "table": "DAILY",
            "indicator": indicator,
            "isfilter": "N",  # SYNTHETIC
            "isprimarykey": "N",  # SYNTHETIC
            "title": indicator,  # SYNTHETIC
            "description": "Synthetic description",  # SYNTHETIC
            "unittype": unit,
        }
        for indicator, unit in units.items()
    ]


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("SHARADAR_API_KEY", "synthetic-key")  # SYNTHETIC


@pytest.fixture
def today(monkeypatch):
    """Set the vendor's "today", which a pull records as its watermark."""

    def set_today(day: str) -> None:
        monkeypatch.setattr(
            "quantlab.acquisition.sharadar.client.vendor_today",
            lambda: date.fromisoformat(day),
        )

    set_today("2024-01-08")
    return set_today


def _vendor(daily, tickers=("AAA",), units=None):
    ticks = [
        tickers_row(label, PERMATICKERS[t], t) for t in tickers for label in ("SEP", "SF1")
    ]
    return FakeVendor(
        {
            "daily": (DAILY_COLUMNS, daily),
            "tickers": (TICKERS_COLUMNS, ticks),
            "descriptions": (INDICATORS_COLUMNS, units or _units()),
        }
    )


def _client(vendor):
    from quantlab.acquisition.sharadar.client import SharadarClient

    return SharadarClient(transport=vendor, sleep=lambda seconds: None)


def _bulk(vendor, download_dir):
    client = _client(vendor)
    for code in ("tickers", "indicators", "daily"):
        client.bulk_table(code, download_dir)
    return download_dir / "sharadar"


def _dataset(tmp_path, root, **fields):
    from quantlab.dataset.config import SharadarDailyConfig
    from quantlab.dataset.sharadar.daily import SharadarDailyDataset

    return SharadarDailyDataset(
        SharadarDailyConfig(
            zarr_file_path=str(tmp_path / "sharadar_daily_1d.zarr"),
            raw_data_dir_path=str(root),
            **fields,
        )
    )


def _panel(tmp_path, root, **fields):
    """Build the store with ``update()`` and read it back."""
    _dataset(tmp_path, root, **fields).update()
    return _dataset(tmp_path, root, **fields).panel("2024-01-01", "2024-12-31").load()


def _days(panel) -> list[str]:
    return pd.DatetimeIndex(panel["timestamp"].values).strftime("%Y-%m-%d").tolist()


def test_market_cap_and_ev_are_in_usd_and_every_variable_records_its_unit(tmp_path, today):
    daily = [
        daily_row("AAA", DAYS[0], marketcap=1234.5, ev=2000.25, pe=18.5, pb=3.0, ps=2.5,
                  evebit=12.0, evebitda=9.0),  # SYNTHETIC
    ]
    panel = _panel(tmp_path, _bulk(_vendor(daily), tmp_path / "dl"))

    row = panel.sel(timestamp=DAYS[0], symbol=101)
    assert float(row["marketcap"]) == 1_234_500_000.0
    assert float(row["ev"]) == 2_000_250_000.0
    assert float(row["pe"]) == 18.5
    assert panel["marketcap"].attrs["unit"] == "USD"
    assert panel["ev"].attrs["unit"] == "USD"
    for name in ("evebit", "evebitda", "pb", "pe", "ps"):
        assert panel[name].attrs["unit"] == "ratio"
    assert sorted(panel.data_vars) == sorted(DAILY_UNITS)


def test_a_vendor_unit_other_than_usd_millions_is_refused(tmp_path, today):
    daily = [daily_row("AAA", DAYS[0], marketcap=1.0, ev=1.0)]  # SYNTHETIC
    root = _bulk(_vendor(daily, units=_units(marketcap="USD")), tmp_path / "dl")

    with pytest.raises(ValueError, match="USD millions"):
        _dataset(tmp_path, root).update()


def test_the_symbol_axis_is_the_permaticker_across_a_ticker_change(tmp_path, today):
    # AAA was renamed AAB on 2024-01-04; both tickers are the one company.
    daily = [
        *(daily_row("AAA", d, marketcap=10.0) for d in DAYS[:2]),  # SYNTHETIC
        *(daily_row("AAB", d, marketcap=11.0) for d in DAYS[2:]),  # SYNTHETIC
        *(daily_row("BBB", d, marketcap=20.0) for d in DAYS),  # SYNTHETIC
    ]
    vendor = _vendor(daily, tickers=("AAA", "BBB"))
    vendor.tables["tickers"][1].append(tickers_row("SF1", 101, "AAB"))  # SYNTHETIC
    panel = _panel(tmp_path, _bulk(vendor, tmp_path / "dl"))

    assert panel["symbol"].dtype == np.int64
    assert panel["symbol"].values.tolist() == [101, 202]
    assert panel["marketcap"].sel(symbol=101).values.tolist() == [
        10e6, 10e6, 11e6, 11e6, 11e6
    ]
    assert _days(panel) == DAYS


def test_the_category_filter_applies_only_to_an_unrostered_universe(tmp_path, today):
    daily = [daily_row(t, DAYS[0], marketcap=1.0) for t in ("AAA", "BBB")]  # SYNTHETIC
    vendor = _vendor(daily, tickers=("AAA", "BBB"))
    for row in vendor.tables["tickers"][1]:
        if row["ticker"] == "BBB":
            row["category"] = "ADR Common Stock"  # SYNTHETIC
    root = _bulk(vendor, tmp_path / "dl")

    assert _panel(tmp_path, root)["symbol"].values.tolist() == [101]
    assert _panel(tmp_path / "r", root, permatickers=(202,))["symbol"].values.tolist() == [202]


def test_two_tickers_of_one_permaticker_on_one_date_are_left_out_and_reported(tmp_path, today):
    # TICKERS gives 101 both tickers: neither row can be chosen.
    daily = [
        daily_row("AAA", DAYS[0], marketcap=1.0),  # SYNTHETIC
        daily_row("AAA.OLD", DAYS[0], marketcap=2.0),  # SYNTHETIC
        daily_row("AAA", DAYS[1], marketcap=3.0),  # SYNTHETIC
    ]
    vendor = _vendor(daily)
    vendor.tables["tickers"][1].append(tickers_row("SF1", 101, "AAA.OLD"))  # SYNTHETIC
    root = _bulk(vendor, tmp_path / "dl")

    panel = _panel(tmp_path, root)
    assert panel["marketcap"].sel(symbol=101).values.tolist() == [3e6]
    report = _ambiguous_report(tmp_path)
    assert sorted(report) == ["AAA", "AAA.OLD"]
    assert "several" in report["AAA"]["reason"]


def _ambiguous_report(tmp_path) -> dict:
    """The ``<store>.unmapped.json`` entries by ticker, of the one store under ``tmp_path``."""
    (path,) = tmp_path.glob("*.unmapped.json")
    return {e["ticker"]: e for e in json.loads(path.read_text())["unmapped"]}


def test_an_update_by_lastupdated_appends_new_days_and_never_rewrites_stored_ones(
    tmp_path, today
):
    daily = [daily_row("AAA", d, marketcap=10.0, pe=5.0) for d in DAYS[:3]]  # SYNTHETIC
    vendor = _vendor(daily, tickers=("AAA", "BBB"))
    root = _bulk(vendor, tmp_path / "dl")
    _dataset(tmp_path, root).update()
    before = _dataset(tmp_path, root).panel("2024-01-01", "2024-12-31").load()

    # The vendor moves on: two new days, a new company, and a silent change
    # to a stored day.
    daily[0] = daily_row("AAA", DAYS[0], marketcap=99.0, pe=5.0, lastupdated="2024-01-08")  # SYNTHETIC
    daily.extend(
        daily_row(t, d, marketcap=12.0, pe=6.0, lastupdated="2024-01-08")  # SYNTHETIC
        for t in ("AAA", "BBB")
        for d in DAYS[3:]
    )
    vendor.calls.clear()
    today("2024-01-09")
    _client(vendor).updated_table("daily", tmp_path / "dl")
    _dataset(tmp_path, root).update()

    after = _dataset(tmp_path, root).panel("2024-01-01", "2024-12-31").load()
    assert all("years" not in c.params for c in vendor.calls)
    assert [c.params["lastupdated.gte"] for c in vendor.calls] == ["2024-01-08"]
    assert _days(after) == DAYS
    assert after["symbol"].values.tolist() == [101, 202]
    assert after.sel(timestamp=slice(None, DAYS[2]), symbol=[101]).identical(before)
    assert after["marketcap"].sel(symbol=101).values.tolist() == [10e6] * 3 + [12e6] * 2
    assert after["marketcap"].attrs["unit"] == "USD"
    assert np.isnan(after["marketcap"].sel(symbol=202).values[:3]).all()
