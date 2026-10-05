"""Sharadar daily update: trailing-window pulls, append-only stores, reported corrections.

A `FakeVendor` plays Sharadar over time: the tests change its rows between
pulls, and the real client, raw tier and dataset run against it. Every row
is invented (`# SYNTHETIC`).
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from tests.sharadar_fixtures import (
    ACTIONS_COLUMNS,
    SEP_COLUMNS,
    TICKERS_COLUMNS,
    FakeVendor,
    action_row,
    sep_row,
    tickers_row,
)

DAYS = [  # Weekdays of January 2024.
    "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
    "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11",
]


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("SHARADAR_API_KEY", "synthetic-key")  # SYNTHETIC


def _vendor(days, tickers=("AAA",)):
    """A vendor with one SEP row per ticker per day (closes 100, 101, ...), no actions."""
    sep = [
        sep_row(t, d, 100.0 + i + 10 * k)  # SYNTHETIC
        for k, t in enumerate(tickers)
        for i, d in enumerate(days)
    ]
    ticks = [tickers_row("SEP", 101 + 101 * k, t) for k, t in enumerate(tickers)]  # SYNTHETIC
    return FakeVendor(
        {
            "stocks": (SEP_COLUMNS, sep),
            "tickers": (TICKERS_COLUMNS, ticks),
            "actions": (ACTIONS_COLUMNS, []),
        }
    )


def _client(vendor, **options):
    from quantlab.acquisition.sharadar.client import SharadarClient

    return SharadarClient(transport=vendor, sleep=lambda seconds: None, **options)


def _bulk(vendor, download_dir):
    client = _client(vendor)
    for code in ("sep", "tickers", "actions"):
        client.bulk_table(code, download_dir)
    return download_dir / "sharadar"


def _dataset(tmp_path, root, name="sharadar_sep_1d.zarr", **fields):
    from quantlab.dataset.config import SharadarDatasetConfig
    from quantlab.dataset.sharadar.stock import SharadarStockDataset

    return SharadarStockDataset(
        SharadarDatasetConfig(
            zarr_file_path=str(tmp_path / name), raw_data_dir_path=str(root), **fields
        )
    )


def _window(vendor, download_dir, through, codes=("sep", "actions"), **options):
    client = _client(vendor)
    for code in codes:
        client.window_table(code, download_dir, through=through, **options)


def _raw(root, code):
    from quantlab.dataset.sharadar.tables import scan_raw_table

    return scan_raw_table(root, code).collect().sort("ticker", "date")


# -- the raw tier ----------------------------------------------------------------


def test_a_window_pull_pages_every_day_and_writes_its_rows(tmp_path):
    vendor = _vendor(DAYS[:4], tickers=("AAA", "BBB", "CCC"))
    root = _bulk(vendor, tmp_path)
    vendor.tables["stocks"][1].extend(
        sep_row(t, DAYS[4], 50.0) for t in ("AAA", "BBB", "CCC")  # SYNTHETIC
    )
    _window(vendor, tmp_path, DAYS[4], codes=("sep",), page_rows=2)

    raw = _raw(root, "sep")
    assert raw.filter(pl.col("date") == pl.date(2024, 1, 8)).height == 3
    assert raw.height == 15  # no row of the overlap is duplicated
    # Three rows on the new day at two a page: offsets 0 and 2.
    offsets = [c["offset"] for c in vendor.window_calls("stocks") if c["from"] == DAYS[4]]
    assert offsets == ["0", "2"]


def test_the_window_starts_a_number_of_trading_days_before_the_watermark(tmp_path):
    from quantlab.dataset.sharadar.tables import read_watermark

    vendor = _vendor(DAYS[:6])
    root = _bulk(vendor, tmp_path)
    _window(vendor, tmp_path, DAYS[7], codes=("sep",), trading_days=3)

    days = sorted({c["from"] for c in vendor.window_calls("stocks")})
    # The third-last raw date through the requested day, every calendar day.
    assert days[0] == DAYS[3]
    assert days[-1] == DAYS[7]
    assert "2024-01-06" in days  # a weekend is asked for, and answers nothing
    assert str(read_watermark(root, "sep")) == DAYS[7]


def test_a_window_replaces_the_rows_of_its_dates_and_keeps_the_rest(tmp_path):
    vendor = _vendor(DAYS[:4])
    root = _bulk(vendor, tmp_path)
    rows = vendor.tables["stocks"][1]
    rows[3]["closeunadj"] = 999.0  # SYNTHETIC: the vendor corrects the last day
    rows[0]["closeunadj"] = 555.0  # SYNTHETIC: outside the window, never re-pulled
    _window(vendor, tmp_path, DAYS[3], codes=("sep",), trading_days=2)

    assert _raw(root, "sep")["closeunadj"].to_list() == [100.0, 101.0, 102.0, 999.0]


def test_a_failed_window_pull_leaves_the_raw_tier_and_watermark_unchanged(tmp_path):
    from quantlab.acquisition.sharadar.client import SharadarHttpError
    from quantlab.dataset.sharadar.tables import read_watermark

    vendor = _vendor(DAYS[:4])
    root = _bulk(vendor, tmp_path)
    before = read_watermark(root, "sep")
    vendor.tables["stocks"][1].append(sep_row("AAA", DAYS[4], 104.0))  # SYNTHETIC
    vendor.fail_on.add(DAYS[4])
    with pytest.raises(SharadarHttpError):
        _window(vendor, tmp_path, DAYS[4], codes=("sep",))
    assert read_watermark(root, "sep") == before
    assert _raw(root, "sep").height == 4
    assert sorted(p.name for p in (root / "sep").iterdir() if p.suffix == ".parquet") == [
        "sep.parquet"
    ]


def test_a_bulk_pull_supersedes_the_windows(tmp_path):
    vendor = _vendor(DAYS[:4])
    root = _bulk(vendor, tmp_path)
    vendor.tables["stocks"][1].append(sep_row("AAA", DAYS[4], 104.0))  # SYNTHETIC
    _window(vendor, tmp_path, DAYS[4], codes=("sep",))
    _client(vendor).bulk_table("sep", tmp_path)
    assert sorted(p.name for p in (root / "sep").glob("*.parquet")) == ["sep.parquet"]
    assert _raw(root, "sep").height == 5


# -- the store ---------------------------------------------------------------------


def _stored(ds, start="2024-01-01", end="2024-12-31"):
    return ds.panel(start, end).load()


def test_an_update_appends_new_bars_and_leaves_earlier_rows_identical(tmp_path):
    vendor = _vendor(DAYS[:4])
    root = _bulk(vendor, tmp_path)
    _dataset(tmp_path, root).update()
    before = _stored(_dataset(tmp_path, root))

    # Two more days, with a $2 dividend on the second.
    vendor.tables["stocks"][1].extend(
        [sep_row("AAA", DAYS[4], 104.0), sep_row("AAA", DAYS[5], 103.0)]  # SYNTHETIC
    )
    vendor.tables["actions"][1].append(action_row(DAYS[5], "dividend", "AAA", 2.0))  # SYNTHETIC
    _window(vendor, tmp_path, DAYS[5])
    ds = _dataset(tmp_path, root)
    ds.update()

    after = _stored(ds)
    assert after.sizes["timestamp"] == 6
    old = after.isel(timestamp=slice(0, 4))
    for name in before.data_vars:
        assert old[name].dtype == before[name].dtype
        np.testing.assert_array_equal(old[name].values, before[name].values)
    # The new bars chain from the stored last adjusted close.
    last = float(before["adjClose"].isel(timestamp=-1, symbol=0))
    expected = [last * 104.0 / 103.0, last * 104.0 / 103.0 * (103.0 + 2.0) / 104.0]
    np.testing.assert_allclose(after["adjClose"].isel(symbol=0).values[4:], expected)
    assert after["divCash"].isel(symbol=0).values[4:].tolist() == [0.0, 2.0]


def test_an_updated_store_equals_a_store_built_from_scratch(tmp_path):
    vendor = _vendor(DAYS[:4], tickers=("AAA", "BBB"))
    root = _bulk(vendor, tmp_path)
    _dataset(tmp_path, root).update()
    vendor.tables["stocks"][1].extend(
        sep_row(t, d, 90.0 + i)  # SYNTHETIC
        for t in ("AAA", "BBB") for i, d in enumerate(DAYS[4:])
    )
    vendor.tables["actions"][1].extend(
        [
            action_row(DAYS[5], "dividend", "BBB", 1.5),  # SYNTHETIC
            action_row(DAYS[6], "split", "AAA", 2.0),  # SYNTHETIC
        ]
    )
    _window(vendor, tmp_path, DAYS[7])
    updated = _dataset(tmp_path, root)
    updated.update()
    _dataset(tmp_path, root, name="fresh.zarr").from_raw_data().save()

    a, b = _stored(updated), _stored(_dataset(tmp_path, root, name="fresh.zarr"))
    for name in ("close", "adjClose", "adjOpen", "adjVolume", "divCash", "splitFactor"):
        np.testing.assert_allclose(a[name].values, b[name].values, rtol=1e-12)


def test_a_security_halted_longer_than_the_overlap_continues_its_chain(tmp_path, monkeypatch):
    from quantlab.dataset.sharadar import stock

    monkeypatch.setattr(stock, "UPDATE_OVERLAP_BARS", 2)
    vendor = _vendor(DAYS[:6], tickers=("AAA", "BBB"))
    rows = vendor.tables["stocks"][1]
    # BBB trades on the first two days only, then is halted.
    rows[:] = [r for r in rows if r["ticker"] == "AAA" or r["date"] <= DAYS[1]]
    root = _bulk(vendor, tmp_path)
    _dataset(tmp_path, root).update()
    rows.extend([sep_row("AAA", DAYS[6], 120.0), sep_row("BBB", DAYS[6], 99.0)])  # SYNTHETIC
    vendor.tables["actions"][1].append(action_row(DAYS[6], "dividend", "BBB", 1.0))  # SYNTHETIC
    _window(vendor, tmp_path, DAYS[6])
    updated = _dataset(tmp_path, root)
    updated.update()
    _dataset(tmp_path, root, name="fresh.zarr").from_raw_data().save()

    a, b = _stored(updated), _stored(_dataset(tmp_path, root, name="fresh.zarr"))
    np.testing.assert_allclose(a["adjClose"].values, b["adjClose"].values, rtol=1e-12)
    bbb = a["adjClose"].sel(symbol=202).values
    assert bbb[-1] == pytest.approx(bbb[1] * (99.0 + 1.0) / 111.0)


def test_a_vendor_correction_to_a_stored_date_is_reported_and_not_written(tmp_path):
    vendor = _vendor(DAYS[:4])
    root = _bulk(vendor, tmp_path)
    _dataset(tmp_path, root).update()
    rows = vendor.tables["stocks"][1]
    rows[3]["closeunadj"] = 103.5  # SYNTHETIC: the vendor corrects a stored day
    rows.append(sep_row("AAA", DAYS[4], 104.0))  # SYNTHETIC
    _window(vendor, tmp_path, DAYS[4])
    ds = _dataset(tmp_path, root)
    ds.update()

    panel = _stored(ds)
    assert float(panel["close"].isel(symbol=0, timestamp=3)) == 103.0
    assert float(panel["close"].isel(symbol=0, timestamp=4)) == 104.0
    report = json.loads(ds.corrections_path().read_text())
    assert {
        "table": "sep",
        "permaticker": 101,
        "date": DAYS[3],
        "variable": "close",
        "stored": 103.0,
        "vendor": 103.5,
    } in report


def test_an_update_stops_at_the_oldest_table_watermark_and_resumes(tmp_path):
    vendor = _vendor(DAYS[:4])
    root = _bulk(vendor, tmp_path)
    _dataset(tmp_path, root).update()
    vendor.tables["stocks"][1].extend(
        [sep_row("AAA", DAYS[4], 104.0), sep_row("AAA", DAYS[5], 105.0)]  # SYNTHETIC
    )
    vendor.tables["actions"][1].append(action_row(DAYS[5], "dividend", "AAA", 1.0))  # SYNTHETIC
    # ACTIONS was refreshed through DAYS[4] only, SEP through DAYS[5]: the
    # dividend on DAYS[5] is not known yet, so DAYS[5] must not be stored.
    _window(vendor, tmp_path, DAYS[4], codes=("actions",))
    _window(vendor, tmp_path, DAYS[5], codes=("sep",))
    _dataset(tmp_path, root).update()
    assert _stored(_dataset(tmp_path, root)).sizes["timestamp"] == 5

    _window(vendor, tmp_path, DAYS[5], codes=("actions",))
    ds = _dataset(tmp_path, root)
    ds.update()
    panel = _stored(ds)
    assert panel.sizes["timestamp"] == 6
    assert panel["divCash"].isel(symbol=0).values[-1] == 1.0


def test_a_new_listing_is_added_with_no_history(tmp_path):
    vendor = _vendor(DAYS[:4])
    root = _bulk(vendor, tmp_path)
    _dataset(tmp_path, root).update()
    vendor.tables["tickers"][1].append(tickers_row("SEP", 909, "NEW"))  # SYNTHETIC
    vendor.tables["stocks"][1].extend(
        [sep_row("AAA", DAYS[4], 104.0), sep_row("NEW", DAYS[4], 20.0)]  # SYNTHETIC
    )
    _client(vendor).bulk_table("tickers", tmp_path)
    _window(vendor, tmp_path, DAYS[4])
    ds = _dataset(tmp_path, root)
    ds.update()
    panel = _stored(ds)
    assert panel.symbol.values.tolist() == [101, 909]
    assert np.isnan(panel["close"].sel(symbol=909).values[:4]).all()
    assert float(panel["close"].sel(symbol=909).values[4]) == 20.0


def test_an_update_with_nothing_new_changes_nothing(tmp_path):
    vendor = _vendor(DAYS[:4])
    root = _bulk(vendor, tmp_path)
    _dataset(tmp_path, root).update()
    before = _stored(_dataset(tmp_path, root))
    _window(vendor, tmp_path, DAYS[3])
    _dataset(tmp_path, root).update()
    after = _stored(_dataset(tmp_path, root))
    assert dict(after.sizes) == dict(before.sizes)
    np.testing.assert_array_equal(after["adjClose"].values, before["adjClose"].values)


def test_without_a_store_update_builds_one(tmp_path):
    vendor = _vendor(DAYS[:4])
    root = _bulk(vendor, tmp_path)
    ds = _dataset(tmp_path, root)
    ds.update()
    assert _stored(ds).sizes["timestamp"] == 4
