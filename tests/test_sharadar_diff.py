"""Sharadar bulk diff: a full pull compared with a store, reported and never written.

A `FakeVendor` plays Sharadar: a store is built from one bulk pull, the vendor
then corrects rows, and a second bulk pull (into a separate download directory,
or over the store's own raw tier) is diffed against the store. Every row is
invented (`# SYNTHETIC`).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

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

DAYS = ["2023-12-28", "2023-12-29", "2024-01-02", "2024-01-03", "2024-01-04"]


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("SHARADAR_API_KEY", "synthetic-key")  # SYNTHETIC


def _vendor(api="stocks", label="SEP"):
    """AAA (101) and BBB (202), one row per day each, no actions."""
    rows = [
        sep_row(t, d, 100.0 + i + 10 * k)  # SYNTHETIC
        for k, t in enumerate(("AAA", "BBB"))
        for i, d in enumerate(DAYS)
    ]
    ticks = [tickers_row(label, 101, "AAA"), tickers_row(label, 202, "BBB")]  # SYNTHETIC
    return FakeVendor(
        {
            api: (SEP_COLUMNS, rows),
            "tickers": (TICKERS_COLUMNS, ticks),
            "actions": (ACTIONS_COLUMNS, []),
        }
    )


def _bulk(vendor, download_dir, code="sep"):
    from quantlab.acquisition.sharadar.client import SharadarClient

    client = SharadarClient(transport=vendor, sleep=lambda seconds: None)
    for table in (code, "tickers", "actions"):
        client.bulk_table(table, download_dir)
    return download_dir / "sharadar"


def _dataset(tmp_path, root, **fields):
    from quantlab.dataset.config import SharadarDatasetConfig
    from quantlab.dataset.sharadar.stock import SharadarStockDataset

    return SharadarStockDataset(
        SharadarDatasetConfig(
            zarr_file_path=str(tmp_path / "sharadar_sep_1d.zarr"),
            raw_data_dir_path=str(root),
            **fields,
        )
    )


def _row(vendor, api, ticker, day):
    return next(r for r in vendor.tables[api][1] if r["ticker"] == ticker and r["date"] == day)


def _correct(vendor, api="stocks"):
    """The vendor revises AAA's 2023-12-29 close, drops BBB's 2024-01-03 bar
    and adds a dividend to AAA on 2024-01-04."""
    row = _row(vendor, api, "AAA", "2023-12-29")
    row.update(close=150.0, closeunadj=150.0)  # SYNTHETIC
    vendor.tables[api][1].remove(_row(vendor, api, "BBB", "2024-01-03"))
    vendor.tables["actions"][1].append(action_row("2024-01-04", "dividend", "AAA", 0.5))  # SYNTHETIC


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        sha.update(str(file.relative_to(path)).encode())
        sha.update(file.read_bytes())
    return sha.hexdigest()


def _keys(differences):
    return {(d["table"], d["permaticker"], d["date"], d["variable"]) for d in differences}


def test_a_diff_names_table_permaticker_date_and_variable(tmp_path):
    vendor = _vendor()
    root = _bulk(vendor, tmp_path / "downloads")
    ds = _dataset(tmp_path, root)
    ds.update()

    _correct(vendor)
    fresh = _bulk(vendor, tmp_path / "bulk_check")
    differences = ds.diff(fresh)

    assert _keys(differences) == {
        # The revised close (split-adjusted and raw moved together, so the
        # raw open, high, low and volume are unchanged).
        ("sep", 101, "2023-12-29", "close"),
        ("sep", 101, "2024-01-04", "divCash"),
        # The dropped bar: every raw variable now has no vendor value.
        *{("sep", 202, "2024-01-03", v) for v in ("open", "high", "low", "close", "volume", "divCash", "splitFactor")},
    }
    close = next(d for d in differences if d["variable"] == "close" and d["permaticker"] == 101)
    assert (close["stored"], close["vendor"]) == (101.0, 150.0)
    dropped = next(d for d in differences if d["permaticker"] == 202 and d["variable"] == "close")
    assert (dropped["stored"], dropped["vendor"]) == (113.0, None)


def test_the_store_and_its_raw_tier_are_unchanged_and_the_report_sits_beside_it(tmp_path):
    vendor = _vendor()
    root = _bulk(vendor, tmp_path / "downloads")
    ds = _dataset(tmp_path, root)
    ds.update()
    store = Path(ds.store_path)
    before = (_digest(store), _digest(root), sorted(p.name for p in tmp_path.iterdir()))

    _correct(vendor)
    differences = ds.diff(_bulk(vendor, tmp_path / "bulk_check"))

    assert _digest(store) == before[0]
    assert _digest(root) == before[1]
    assert not ds.corrections_path().exists()
    report = json.loads(ds.diff_path().read_text())
    assert report["differences"] == differences
    assert report["table"] == "sep"
    assert (report["from"], report["to"]) == ("2023-12-28", "2024-01-04")


def test_an_unchanged_vendor_diffs_to_nothing(tmp_path):
    vendor = _vendor()
    root = _bulk(vendor, tmp_path / "downloads")
    ds = _dataset(tmp_path, root)
    ds.update()
    assert ds.diff(_bulk(vendor, tmp_path / "bulk_check")) == []
    assert json.loads(ds.diff_path().read_text())["differences"] == []


def test_by_default_the_diff_reads_the_stores_own_raw_tier(tmp_path):
    # A bulk pull over the store's own download directory replaces its raw
    # tier; the store keeps what it held, and the diff shows the gap.
    vendor = _vendor()
    root = _bulk(vendor, tmp_path / "downloads")
    ds = _dataset(tmp_path, root)
    ds.update()
    _correct(vendor)
    _bulk(vendor, tmp_path / "downloads")
    assert ("sep", 101, "2023-12-29", "close") in _keys(ds.diff())


def test_a_fund_store_diffs_through_its_own_table(tmp_path):
    vendor = _vendor(api="funds", label="SFP")
    root = _bulk(vendor, tmp_path / "downloads", code="sfp")
    ds = _dataset(tmp_path, root, table="sfp")
    ds.update()
    _correct(vendor, api="funds")
    differences = ds.diff(_bulk(vendor, tmp_path / "bulk_check", code="sfp"))
    assert ("sfp", 101, "2023-12-29", "close") in _keys(differences)


def test_a_diff_without_a_store_is_refused(tmp_path):
    root = _bulk(_vendor(), tmp_path / "downloads")
    with pytest.raises(FileNotFoundError):
        _dataset(tmp_path, root).diff()
