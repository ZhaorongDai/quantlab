"""Sharadar SF1 fundamentals: a point-in-time panel at release dates, refreshed by lastupdated.

A `FakeVendor` plays Sharadar; the real client, raw tier and dataset run
against it. SF1's columns are VERBATIM, every row value is invented
(`# SYNTHETIC`). The panel's calendar is SEP's trading days, here the
weekdays of January 2024.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr

from tests.sharadar_fixtures import (
    DAILY_COLUMNS,
    daily_row,
    INDICATORS_COLUMNS,
    SEP_COLUMNS,
    SF1_COLUMNS,
    TICKERS_COLUMNS,
    FakeVendor,
    sep_row,
    sf1_row,
    tickers_row,
)

DAYS = [  # Weekdays of January 2024.
    "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
    "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11",
    "2024-01-12", "2024-01-15", "2024-01-16", "2024-01-17",
]
PERMATICKERS = {"AAA": 101, "BBB": 202, "CCC": 303}  # SYNTHETIC

UNITS = [
    {
        "table": "SF1",
        "indicator": indicator,
        "isfilter": "N",  # SYNTHETIC
        "isprimarykey": "N",  # SYNTHETIC
        "title": indicator,  # SYNTHETIC
        "description": "Synthetic description",  # SYNTHETIC
        "unittype": unit,  # VERBATIM unit types of SF1 indicators
    }
    for indicator, unit in (("revenue", "currency"), ("pe", "ratio"))
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

    set_today("2024-01-11")
    return set_today


def _vendor(sf1, days=DAYS[:8], tickers=("AAA",)):
    sep = [sep_row(t, d, 100.0) for t in tickers for d in days]  # SYNTHETIC
    ticks = [
        tickers_row(label, PERMATICKERS[t], t)
        for t in tickers
        for label in ("SEP", "SF1")
    ]
    return FakeVendor(
        {
            "stocks": (SEP_COLUMNS, sep),
            "tickers": (TICKERS_COLUMNS, ticks),
            "fundamentals": (SF1_COLUMNS, sf1),
            "descriptions": (INDICATORS_COLUMNS, UNITS),
        }
    )


def _client(vendor, **options):
    from quantlab.acquisition.sharadar.client import SharadarClient

    return SharadarClient(transport=vendor, sleep=lambda seconds: None, **options)


def _bulk(vendor, download_dir):
    client = _client(vendor)
    for code in ("tickers", "indicators", "sep", "sf1"):
        client.bulk_table(code, download_dir)
    return download_dir / "sharadar"


def _dataset(tmp_path, root, **fields):
    from quantlab.dataset.config import SharadarFundamentalsConfig
    from quantlab.dataset.sharadar.fundamentals import SharadarFundamentalsDataset

    name = f"sharadar_sf1_{fields.get('dimension', 'ARQ').lower()}.zarr"
    return SharadarFundamentalsDataset(
        SharadarFundamentalsConfig(
            zarr_file_path=str(tmp_path / name), raw_data_dir_path=str(root), **fields
        )
    )


def _panel(tmp_path, root, **fields):
    """Build the store with ``update()`` and read it back."""
    _dataset(tmp_path, root, **fields).update()
    return _dataset(tmp_path, root, **fields).panel("2024-01-01", "2024-12-31")


def _series(panel, variable, ticker="AAA") -> dict[str, float]:
    """``{day: value}`` of one security's variable, NaN kept."""
    column = panel[variable].sel(symbol=PERMATICKERS[ticker])
    stamps = pd.DatetimeIndex(panel["timestamp"].values).strftime("%Y-%m-%d")
    return dict(zip(stamps, column.values.tolist()))


def _shown(series: dict[str, float]) -> dict[str, float]:
    return {day: value for day, value in series.items() if not np.isnan(value)}


# -- placing rows at their release date --------------------------------------


def test_a_row_is_shown_from_its_release_date_until_a_newer_period_replaces_it(
    tmp_path, today
):
    sf1 = [
        sf1_row("AAA", "ARQ", "2024-01-04", "2023-09-30", revenue=100),  # SYNTHETIC
        sf1_row("AAA", "ARQ", "2024-01-09", "2023-12-31", revenue=200),  # SYNTHETIC
    ]
    panel = _panel(tmp_path, _bulk(_vendor(sf1), tmp_path / "dl"))

    assert pd.DatetimeIndex(panel["timestamp"].values).strftime("%Y-%m-%d").tolist() == DAYS[:8]
    assert panel["symbol"].values.tolist() == [101]
    assert _shown(_series(panel, "revenue")) == {
        "2024-01-04": 100.0, "2024-01-05": 100.0, "2024-01-08": 100.0,
        "2024-01-09": 200.0, "2024-01-10": 200.0, "2024-01-11": 200.0,
    }
    release = panel["release_date"].sel(symbol=101).values
    period = panel["reportperiod"].sel(symbol=101).values
    assert pd.Timestamp(release[3]) == pd.Timestamp("2024-01-04")
    assert pd.Timestamp(period[3]) == pd.Timestamp("2023-09-30")
    assert pd.Timestamp(release[-1]) == pd.Timestamp("2024-01-09")
    assert pd.Timestamp(period[-1]) == pd.Timestamp("2023-12-31")
    assert np.isnat(release[0])


def test_a_weekend_release_is_shown_from_the_next_trading_day(tmp_path, today):
    sf1 = [sf1_row("AAA", "ARQ", "2024-01-06", "2023-12-31", revenue=100)]  # SYNTHETIC
    panel = _panel(tmp_path, _bulk(_vendor(sf1), tmp_path / "dl"))

    assert min(_shown(_series(panel, "revenue"))) == "2024-01-08"


def test_no_most_recent_dimension_row_reaches_the_store(tmp_path, today):
    # MRQ is dated at its period end, three weeks before the period's filing:
    # it would leak the value early.
    sf1 = [
        sf1_row("AAA", "MRQ", "2023-12-31", "2023-12-31", revenue=999),  # SYNTHETIC
        sf1_row("AAA", "MRT", "2023-12-31", "2023-12-31", revenue=998),  # SYNTHETIC
        sf1_row("AAA", "MRY", "2023-12-31", "2023-12-31", revenue=997),  # SYNTHETIC
        sf1_row("AAA", "ARQ", "2024-01-09", "2023-12-31", revenue=200),  # SYNTHETIC
        sf1_row("AAA", "ART", "2024-01-09", "2023-12-31", revenue=800),  # SYNTHETIC
    ]
    root = _bulk(_vendor(sf1), tmp_path / "dl")

    arq = _panel(tmp_path, root, dimension="ARQ")
    art = _panel(tmp_path, root, dimension="ART")

    assert _shown(_series(arq, "revenue")) == {d: 200.0 for d in DAYS[5:8]}
    assert _shown(_series(art, "revenue")) == {d: 800.0 for d in DAYS[5:8]}


@pytest.mark.parametrize("dimension", ["MRQ", "MRY", "MRT", "ARY", "XYZ"])
def test_a_store_holds_only_an_as_reported_quarterly_or_trailing_dimension(
    tmp_path, dimension
):
    with pytest.raises(ValueError, match="ARQ"):
        _dataset(tmp_path, tmp_path, dimension=dimension)


def test_a_restatement_is_shown_from_its_own_release_date_never_earlier(tmp_path, today):
    sf1 = [
        sf1_row("AAA", "ARQ", "2024-01-04", "2023-09-30", revenue=100),  # SYNTHETIC
        # The same quarter refiled a week later with a corrected value.
        sf1_row("AAA", "ARQ", "2024-01-10", "2023-09-30", revenue=110),  # SYNTHETIC
    ]
    panel = _panel(tmp_path, _bulk(_vendor(sf1), tmp_path / "dl"))

    assert _shown(_series(panel, "revenue")) == {
        "2024-01-04": 100.0, "2024-01-05": 100.0, "2024-01-08": 100.0,
        "2024-01-09": 100.0, "2024-01-10": 110.0, "2024-01-11": 110.0,
    }


def test_a_later_filing_of_an_older_period_never_replaces_a_newer_period(tmp_path, today):
    sf1 = [
        sf1_row("AAA", "ARQ", "2024-01-04", "2023-12-31", revenue=200),  # SYNTHETIC
        # An amendment of the quarter before, filed after the newer quarter.
        sf1_row("AAA", "ARQ", "2024-01-09", "2023-09-30", revenue=111),  # SYNTHETIC
    ]
    panel = _panel(tmp_path, _bulk(_vendor(sf1), tmp_path / "dl"))

    assert set(_shown(_series(panel, "revenue")).values()) == {200.0}


def test_a_row_not_replaced_goes_stale(tmp_path, today):
    sf1 = [sf1_row("AAA", "ARQ", "2024-01-04", "2023-09-30", revenue=100)]  # SYNTHETIC
    root = _bulk(_vendor(sf1), tmp_path / "dl")

    stale = _panel(tmp_path, root, stale_after_days=3)
    forever = _panel(tmp_path / "forever", root, stale_after_days=None)

    # Shown on its release day and the three days after it (a weekend).
    assert list(_shown(_series(stale, "revenue"))) == ["2024-01-04", "2024-01-05"]
    assert list(_shown(_series(forever, "revenue"))) == DAYS[2:8]


def test_each_indicator_records_its_vendor_unit(tmp_path, today):
    sf1 = [sf1_row("AAA", "ARQ", "2024-01-04", "2023-09-30", revenue=100, pe=12.5)]  # SYNTHETIC
    panel = _panel(tmp_path, _bulk(_vendor(sf1), tmp_path / "dl"))

    assert panel["revenue"].attrs["unit"] == "currency"
    assert panel["pe"].attrs["unit"] == "ratio"
    assert _shown(_series(panel, "pe")) == {d: 12.5 for d in DAYS[2:8]}


def test_the_universe_rules_are_the_price_panels(tmp_path, today):
    sf1 = [
        sf1_row(t, "ARQ", "2024-01-04", "2023-09-30", revenue=100)  # SYNTHETIC
        for t in ("AAA", "BBB")
    ]
    vendor = _vendor(sf1, tickers=("AAA", "BBB"))
    columns, rows = vendor.tables["tickers"]
    for row in rows:
        if row["ticker"] == "BBB":
            row["category"] = "ADR Common Stock"  # SYNTHETIC
    root = _bulk(vendor, tmp_path / "dl")

    default = _panel(tmp_path, root)
    rostered = _panel(tmp_path / "rostered", root, permatickers=(202,))

    assert default["symbol"].values.tolist() == [101]
    assert rostered["symbol"].values.tolist() == [202]


# -- refreshing by lastupdated ---------------------------------------------


def test_an_updated_pull_asks_for_rows_changed_since_the_watermark(tmp_path, today):
    sf1 = [sf1_row("AAA", "ARQ", "2024-01-04", "2023-09-30", revenue=100)]  # SYNTHETIC
    vendor = _vendor(sf1)
    root = _bulk(vendor, tmp_path / "dl")
    sf1.append(
        sf1_row("AAA", "ARQ", "2024-01-10", "2023-12-31", revenue=200, lastupdated="2024-01-11")  # SYNTHETIC
    )
    vendor.calls.clear()

    today("2024-01-12")
    _client(vendor).updated_table("sf1", tmp_path / "dl")

    requests = [c.params for c in vendor.calls]
    assert all("years" not in params for params in requests)
    assert requests[0]["lastupdated.gte"] == "2024-01-11"
    from quantlab.dataset.sharadar.tables import read_watermark, scan_raw_table

    assert read_watermark(root, "sf1") == date(2024, 1, 12)
    raw = scan_raw_table(root, "sf1").collect()
    assert sorted(raw.get_column("revenue").to_list()) == [100, 200]


def test_an_updated_pull_pages_at_whole_tickers(tmp_path, today):
    sf1 = [
        sf1_row(t, dim, "2024-01-04", "2023-09-30", revenue=100 + k, lastupdated="2024-01-11")  # SYNTHETIC
        for k, t in enumerate(("AAA", "BBB", "CCC"))
        for dim in ("ARQ", "ART")
    ]
    vendor = _vendor([], tickers=("AAA", "BBB", "CCC"))
    root = _bulk(vendor, tmp_path / "dl")
    vendor.tables["fundamentals"] = (SF1_COLUMNS, sf1)

    _client(vendor).updated_table("sf1", tmp_path / "dl", page_rows=3)

    from quantlab.dataset.sharadar.tables import scan_raw_table

    raw = scan_raw_table(root, "sf1").collect()
    assert raw.height == 6
    assert raw.select("ticker", "dimension").unique().height == 6
    cursors = [c.params.get("ticker.gte") for c in vendor.calls if "lastupdated.gte" in c.params]
    assert cursors == [None, "BBB", "CCC"]


def test_an_updated_row_replaces_the_row_with_its_key_and_a_bulk_pull_supersedes_both(
    tmp_path, today
):
    sf1 = [sf1_row("AAA", "ARQ", "2024-01-04", "2023-09-30", revenue=100)]  # SYNTHETIC
    vendor = _vendor(sf1)
    root = _bulk(vendor, tmp_path / "dl")
    sf1[0] = sf1_row("AAA", "ARQ", "2024-01-04", "2023-09-30", revenue=105, lastupdated="2024-01-11")  # SYNTHETIC

    _client(vendor).updated_table("sf1", tmp_path / "dl")

    from quantlab.dataset.sharadar.tables import scan_raw_table

    assert scan_raw_table(root, "sf1").collect().get_column("revenue").to_list() == [105]
    _client(vendor).bulk_table("sf1", tmp_path / "dl")
    assert not list((root / "sf1").glob("updated_*"))
    assert scan_raw_table(root, "sf1").collect().get_column("revenue").to_list() == [105]


def test_a_table_without_a_primary_key_is_not_refreshed_by_lastupdated(tmp_path, today):
    root = _bulk(_vendor([]), tmp_path / "dl")

    with pytest.raises(ValueError, match="lastupdated"):
        _client(FakeVendor({})).updated_table("sep", root.parent)


def test_an_update_appends_new_filings_and_never_rewrites_stored_days(tmp_path, today):
    sf1 = [sf1_row("AAA", "ARQ", "2024-01-04", "2023-09-30", revenue=100)]  # SYNTHETIC
    vendor = _vendor(sf1, days=DAYS)
    columns, sep = vendor.tables["stocks"]
    vendor.tables["stocks"] = (columns, [r for r in sep if r["date"] <= "2024-01-11"])
    root = _bulk(vendor, tmp_path / "dl")
    _dataset(tmp_path, root).update()
    before = _dataset(tmp_path, root).panel("2024-01-01", "2024-01-11").load()

    # The vendor moves on: new trading days, a new filing, and a silent change
    # to the stored filing's value.
    vendor.tables["stocks"] = (columns, sep)
    sf1[0] = sf1_row("AAA", "ARQ", "2024-01-04", "2023-09-30", revenue=105, lastupdated="2024-01-15")  # SYNTHETIC
    sf1.append(
        sf1_row("AAA", "ARQ", "2024-01-16", "2023-12-31", revenue=200, lastupdated="2024-01-16")  # SYNTHETIC
    )
    vendor.calls.clear()
    today("2024-01-17")
    client = _client(vendor)
    client.window_table("sep", tmp_path / "dl", through="2024-01-17")
    client.updated_table("sf1", tmp_path / "dl")
    _dataset(tmp_path, root).update()

    after = _dataset(tmp_path, root).panel("2024-01-01", "2024-12-31").load()
    assert not any("years" in c.params for c in vendor.calls)
    assert pd.DatetimeIndex(after["timestamp"].values).strftime("%Y-%m-%d").tolist() == DAYS
    assert after.sel(timestamp=slice(None, "2024-01-11")).identical(before)
    shown = _shown(_series(after, "revenue"))
    assert shown["2024-01-11"] == 100.0
    assert shown["2024-01-15"] == 105.0
    assert shown["2024-01-16"] == 200.0 and shown["2024-01-17"] == 200.0


def test_a_new_filer_widens_the_store(tmp_path, today):
    vendor = _vendor(
        [sf1_row("AAA", "ARQ", "2024-01-04", "2023-09-30", revenue=100)],  # SYNTHETIC
        days=DAYS, tickers=("AAA", "BBB"),
    )
    columns, sep = vendor.tables["stocks"]
    vendor.tables["stocks"] = (columns, [r for r in sep if r["date"] <= "2024-01-11"])
    root = _bulk(vendor, tmp_path / "dl")
    _dataset(tmp_path, root).update()

    vendor.tables["stocks"] = (columns, sep)
    vendor.tables["fundamentals"][1].append(
        sf1_row("BBB", "ARQ", "2024-01-16", "2023-12-31", revenue=7, lastupdated="2024-01-16")  # SYNTHETIC
    )
    today("2024-01-17")
    client = _client(vendor)
    client.window_table("sep", tmp_path / "dl", through="2024-01-17")
    client.updated_table("sf1", tmp_path / "dl")
    _dataset(tmp_path, root).update()

    after = _dataset(tmp_path, root).panel("2024-01-01", "2024-12-31")
    assert after["symbol"].values.tolist() == [101, 202]
    assert _shown(_series(after, "revenue", "BBB")) == {"2024-01-16": 7.0, "2024-01-17": 7.0}


def test_rows_of_one_permaticker_with_one_key_are_refused(tmp_path, today):
    sf1 = [
        sf1_row("AAA", "ARQ", "2024-01-04", "2023-09-30", revenue=100),  # SYNTHETIC
        sf1_row("AAA.OLD", "ARQ", "2024-01-04", "2023-09-30", revenue=101),  # SYNTHETIC
    ]
    vendor = _vendor(sf1)
    vendor.tables["tickers"][1].append(tickers_row("SF1", 101, "AAA.OLD"))  # SYNTHETIC
    root = _bulk(vendor, tmp_path / "dl")

    with pytest.raises(ValueError, match="several raw rows"):
        _dataset(tmp_path, root).update()


def test_raw_rows_keep_every_dimension(tmp_path, today):
    sf1 = [
        sf1_row("AAA", dim, "2024-01-04", "2023-09-30", revenue=100)  # SYNTHETIC
        for dim in ("ARQ", "ART", "ARY", "MRQ", "MRT", "MRY")
    ]
    root = _bulk(_vendor(sf1), tmp_path / "dl")

    from quantlab.dataset.sharadar.tables import scan_raw_table

    raw = scan_raw_table(root, "sf1").collect()
    assert sorted(raw.get_column("dimension").to_list()) == sorted(
        ["ARQ", "ART", "ARY", "MRQ", "MRT", "MRY"]
    )
    assert raw.schema["revenue"] == pl.Int64


def test_a_merge_with_daily_keeps_daily_valuations_under_the_plain_names(tmp_path, today):
    from quantlab.dataset.config import SharadarDailyConfig
    from quantlab.dataset.merged import MergedDataset
    from quantlab.dataset.sharadar.daily import SharadarDailyDataset

    sf1 = [sf1_row("AAA", "ART", "2024-01-04", "2023-09-30", revenue=100, marketcap=5)]  # SYNTHETIC
    vendor = _vendor(sf1)
    vendor.tables["daily"] = (DAILY_COLUMNS, [daily_row("AAA", d, marketcap=7.0) for d in DAYS[:8]])  # SYNTHETIC
    vendor.tables["descriptions"][1].extend(
        {**UNITS[0], "table": "DAILY", "indicator": name, "unittype": "USD millions"}  # VERBATIM unit type
        for name in ("marketcap", "ev")
    )
    root = _bulk(vendor, tmp_path / "dl")
    _client(vendor).bulk_table("daily", tmp_path / "dl")
    art = _dataset(tmp_path, root, dimension="ART").update()
    daily = SharadarDailyDataset(SharadarDailyConfig(
        zarr_file_path=str(tmp_path / "daily.zarr"), raw_data_dir_path=str(root),
    )).update()

    merged = MergedDataset([art, daily]).panel(
        "2024-01-02", "2024-01-11", variables=["marketcap", "sf1_marketcap", "revenue"]
    )

    assert merged["symbol"].values.tolist() == [101]
    # DAILY is in USD millions; the panel holds USD.
    assert set(merged["marketcap"].values.ravel()) == {7.0e6}
    assert merged["sf1_marketcap"].sel(timestamp="2024-01-05").values.tolist() == [5.0]
    assert merged["revenue"].sel(timestamp="2024-01-05").values.tolist() == [100.0]
    assert "marketcap" in art.panel("2024-01-02", "2024-01-11").data_vars
