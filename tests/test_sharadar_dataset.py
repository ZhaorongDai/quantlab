"""The Sharadar SEP panel: raw OHLCV on the permaticker axis.

The raw tier is written by the real client from a faked transport
(`tests/sharadar_fixtures.FakeTransport`), so these tests read exactly what a
bulk pull leaves on disk.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tests.sharadar_fixtures import (
    ACTIONS_COLUMNS,
    SEP_COLUMNS,
    TICKERS_COLUMNS,
    FakeTransport,
    bulk_routes,
    csv_text,
    sep_row,
    tickers_row,
)


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("SHARADAR_API_KEY", "synthetic-key")  # SYNTHETIC


def _pull(download_dir, sep_rows, tickers_rows, actions_rows=()):
    """Pull SEP, TICKERS and ACTIONS through the client; return the vendor root."""
    from quantlab.acquisition.sharadar.client import SharadarClient

    transport = FakeTransport(
        bulk_routes(
            {
                "stocks": csv_text(SEP_COLUMNS, sep_rows),
                "tickers": csv_text(TICKERS_COLUMNS, tickers_rows),
                "actions": csv_text(ACTIONS_COLUMNS, list(actions_rows)),
            }
        )
    )
    client = SharadarClient(transport=transport, sleep=lambda seconds: None)
    for code in ("sep", "tickers", "actions"):
        client.bulk_table(code, download_dir)
    return download_dir / "sharadar"


def _dataset(tmp_path, vendor_root, **fields):
    from quantlab.dataset.config import SharadarDatasetConfig
    from quantlab.dataset.sharadar.stock import SharadarStockDataset

    config = SharadarDatasetConfig(
        zarr_file_path=str(tmp_path / "sharadar_sep_1d.zarr"),
        raw_data_dir_path=str(vendor_root),
        **fields,
    )
    return SharadarStockDataset(config)


def _build(tmp_path, sep_rows, tickers_rows, actions_rows=(), **fields):
    vendor_root = _pull(tmp_path / "downloads", sep_rows, tickers_rows, actions_rows)
    _dataset(tmp_path, vendor_root, **fields).from_raw_data().save()
    return _dataset(tmp_path, vendor_root, **fields)


def test_the_panel_is_raw_ohlcv_on_the_permaticker_axis(tmp_path):
    # A 2:1 split after these dates: SEP's split-adjusted columns are half the
    # raw price, and its volume is double. The panel holds the raw values.
    rows = [
        sep_row(
            "AAA", "2024-01-02", 50.0,  # SYNTHETIC
            open=49.0, high=52.0, low=48.0, volume=2000.0, closeunadj=100.0,  # SYNTHETIC
        ),
        sep_row("AAA", "2024-01-03", 51.0, closeunadj=102.0),  # SYNTHETIC
        sep_row("BBB", "2024-01-02", 20.0),  # SYNTHETIC
    ]
    tickers = [tickers_row("SEP", 101, "AAA"), tickers_row("SEP", 202, "BBB")]  # SYNTHETIC
    ds = _build(tmp_path, rows, tickers)

    panel = ds.panel("2024-01-01", "2024-01-31")
    assert panel.symbol.values.tolist() == [101, 202]
    assert {"open", "high", "low", "close", "volume"} <= set(panel.data_vars)
    # Sharadar's adjusted columns never enter the store.
    assert not {"closeadj", "closeunadj", "lastupdated"} & set(panel.data_vars)

    aaa = panel.sel(symbol=101, timestamp="2024-01-02")
    assert float(aaa["close"]) == 100.0
    assert float(aaa["open"]) == pytest.approx(98.0)
    assert float(aaa["high"]) == pytest.approx(104.0)
    assert float(aaa["low"]) == pytest.approx(96.0)
    assert float(aaa["volume"]) == pytest.approx(1000.0)
    assert float(panel["close"].sel(symbol=202, timestamp="2024-01-03").isnull()) == 1.0


def test_a_ticker_change_keeps_one_permatickers_history_whole(tmp_path):
    # FB became META on 2024-01-03. Sharadar rewires the whole history to the
    # current ticker, and TICKERS keeps the old one only in `relatedtickers`.
    rows = [
        sep_row("META", "2024-01-02", 10.0),  # SYNTHETIC
        sep_row("META", "2024-01-03", 11.0),  # SYNTHETIC
    ]
    tickers = [tickers_row("SEP", 303, "META", relatedtickers="FB")]  # SYNTHETIC
    panel = _build(tmp_path, rows, tickers).panel("2024-01-01", "2024-01-31")
    assert panel.symbol.values.tolist() == [303]
    assert panel["close"].sel(symbol=303).values.tolist() == [10.0, 11.0]


def test_two_tickers_of_one_permaticker_share_its_column(tmp_path):
    # Defensive: should TICKERS ever list an old ticker beside the new one,
    # rows under either still land on the one permaticker.
    rows = [
        sep_row("FB", "2024-01-02", 10.0),  # SYNTHETIC
        sep_row("META", "2024-01-03", 11.0),  # SYNTHETIC
    ]
    tickers = [
        tickers_row("SEP", 303, "FB"),  # SYNTHETIC
        tickers_row("SEP", 303, "META"),  # SYNTHETIC
    ]
    panel = _build(tmp_path, rows, tickers).panel("2024-01-01", "2024-01-31")
    assert panel["close"].sel(symbol=303).values.tolist() == [10.0, 11.0]


def test_a_reused_ticker_never_splices_two_companies(tmp_path):
    # The delisted company's history sits under the suffixed ticker; the new
    # company owns the bare one. Two permatickers, two columns.
    rows = [
        sep_row("ABC1", "2024-01-02", 10.0),  # SYNTHETIC
        sep_row("ABC", "2024-01-04", 70.0),  # SYNTHETIC
    ]
    tickers = [
        tickers_row("SEP", 404, "ABC1", isdelisted="Y"),  # SYNTHETIC
        tickers_row("SEP", 505, "ABC"),  # SYNTHETIC
    ]
    panel = _build(tmp_path, rows, tickers).panel("2024-01-01", "2024-01-31")
    close = panel["close"].to_pandas()
    assert close[404].dropna().index.tolist() == [pd.Timestamp("2024-01-02")]
    assert close[505].dropna().index.tolist() == [pd.Timestamp("2024-01-04")]


def test_the_mapping_uses_only_the_tables_own_tickers_rows(tmp_path):
    # The same ticker under another table (a fund) is a different security.
    rows = [sep_row("XYZ", "2024-01-02", 10.0)]  # SYNTHETIC
    tickers = [
        tickers_row("SFP", 909, "XYZ"),  # SYNTHETIC
        tickers_row("SEP", 606, "XYZ"),  # SYNTHETIC
    ]
    panel = _build(tmp_path, rows, tickers).panel("2024-01-01", "2024-01-31")
    assert panel.symbol.values.tolist() == [606]


def test_a_ticker_mapped_to_two_permatickers_is_refused(tmp_path):
    rows = [sep_row("DUP", "2024-01-02", 10.0)]  # SYNTHETIC
    tickers = [tickers_row("SEP", 1, "DUP"), tickers_row("SEP", 2, "DUP")]  # SYNTHETIC
    vendor_root = _pull(tmp_path / "downloads", rows, tickers)
    with pytest.raises(ValueError, match="DUP"):
        _dataset(tmp_path, vendor_root).from_raw_data()


def test_a_ticker_no_source_maps_is_left_out_and_reported(tmp_path):
    # Not refused (#234): a daily update must complete; the bars are missing
    # and listed in <store>.unmapped.json.
    rows = [
        sep_row("AAA", "2024-01-02", 10.0),  # SYNTHETIC
        sep_row("ZZZ", "2024-01-02", 5.0),  # SYNTHETIC
        sep_row("ZZZ", "2024-01-03", 6.0),  # SYNTHETIC
    ]
    tickers = [tickers_row("SEP", 101, "AAA")]  # SYNTHETIC
    vendor_root = _pull(tmp_path / "downloads", rows, tickers)
    ds = _dataset(tmp_path, vendor_root)
    ds.from_raw_data().save()
    assert ds.panel("2024-01-01", "2024-01-31").symbol.values.tolist() == [101]
    report = json.loads(ds.unmapped_path().read_text())
    assert report["table"] == "sep"
    assert report["unmapped"] == [
        {
            "ticker": "ZZZ",
            "first_date": "2024-01-02",
            "last_date": "2024-01-03",
            "rows": 2,
            "raw_files": ["sep.parquet"],
            "reason": "no permaticker",
        }
    ]


def test_two_tickers_of_one_permaticker_on_one_date_are_left_out_and_reported(tmp_path):
    # TICKERS gives 303 both tickers: neither price can be chosen.
    rows = [
        sep_row("FB", "2024-01-02", 10.0),  # SYNTHETIC
        sep_row("META", "2024-01-02", 10.5),  # SYNTHETIC
        sep_row("META", "2024-01-03", 11.0),  # SYNTHETIC
    ]
    tickers = [tickers_row("SEP", 303, "FB"), tickers_row("SEP", 303, "META")]  # SYNTHETIC
    panel = _build(tmp_path, rows, tickers).panel("2024-01-01", "2024-01-31")
    assert panel["close"].sel(symbol=303).values.tolist() == [11.0]
    (path,) = tmp_path.glob("*.unmapped.json")
    entries = {e["ticker"]: e for e in json.loads(path.read_text())["unmapped"]}
    assert sorted(entries) == ["FB", "META"]
    assert "303" in entries["FB"]["reason"]


def test_the_config_dates_and_permatickers_bound_the_conversion(tmp_path):
    rows = [
        sep_row("AAA", "2024-01-02", 10.0),  # SYNTHETIC
        sep_row("AAA", "2024-01-03", 11.0),  # SYNTHETIC
        sep_row("BBB", "2024-01-03", 20.0),  # SYNTHETIC
    ]
    tickers = [tickers_row("SEP", 101, "AAA"), tickers_row("SEP", 202, "BBB")]  # SYNTHETIC
    ds = _build(
        tmp_path, rows, tickers,
        start_date="2024-01-03", end_date="2024-01-31", permatickers=(101,),
    )
    panel = ds.panel("2024-01-01", "2024-01-31")
    assert panel.symbol.values.tolist() == [101]
    assert panel["close"].values.ravel().tolist() == [11.0]


def test_ticker_symbols_are_refused_in_favour_of_permatickers(tmp_path):
    with pytest.raises(ValueError, match="permatickers"):
        _dataset(tmp_path, tmp_path, symbols=("AAA",))


def test_the_panel_exports_to_kunquant(tmp_path):
    rows = [sep_row("AAA", "2024-01-02", 10.0), sep_row("AAA", "2024-01-03", 11.0)]  # SYNTHETIC
    ds = _build(tmp_path, rows, [tickers_row("SEP", 101, "AAA")])  # SYNTHETIC
    inputs, symbols, _ = ds.to_kunquant(
        ("close", "volume"), panel=ds.panel("2024-01-01", "2024-01-31")
    )
    assert inputs["close"].shape == (2, 1)
    assert inputs["close"].dtype == np.float32
    assert symbols.tolist() == [101]


def test_chunked_conversion_matches_the_one_shot_store(tmp_path):
    from quantlab.dataset.config import SharadarDatasetConfig
    from quantlab.dataset.sharadar.stock import SharadarStockDataset

    rows = [
        sep_row("AAA", "2023-12-29", 9.0),  # SYNTHETIC
        sep_row("AAA", "2024-01-02", 10.0),  # SYNTHETIC
        sep_row("BBB", "2024-01-03", 20.0),  # SYNTHETIC
    ]
    tickers = [tickers_row("SEP", 101, "AAA"), tickers_row("SEP", 202, "BBB")]  # SYNTHETIC
    one_shot = _build(tmp_path, rows, tickers)
    chunked = SharadarStockDataset(
        SharadarDatasetConfig(
            zarr_file_path=str(tmp_path / "chunked.zarr"),
            raw_data_dir_path=str(tmp_path / "downloads" / "sharadar"),
        )
    )
    chunked.from_raw_data_chunked(granularity="year")
    a = one_shot.panel("2023-01-01", "2024-12-31")
    b = chunked.panel("2023-01-01", "2024-12-31")
    assert a.symbol.values.tolist() == b.symbol.values.tolist()
    np.testing.assert_array_equal(a["close"].values, b["close"].values)


def test_the_dataset_rebuilds_from_its_saved_config(tmp_path):
    import json

    from quantlab.core.component import config_to_dict, rebuild

    rows = [sep_row("AAA", "2024-01-02", 10.0)]  # SYNTHETIC
    ds = _build(tmp_path, rows, [tickers_row("SEP", 101, "AAA")], permatickers=(101,))  # SYNTHETIC
    saved = json.loads(json.dumps(config_to_dict(ds.config)))
    rebuilt = rebuild(saved)
    assert type(rebuilt).__name__ == "SharadarStockDataset"
    assert rebuilt.config == ds.config
    assert rebuilt.panel("2024-01-01", "2024-01-31").symbol.values.tolist() == [101]


def test_the_rest_apis_table_name_maps_like_the_bulk_code(tmp_path):
    # The bulk TICKERS file labels SEP rows `SEP`; the REST API labels them
    # `stocks`. Either spelling maps the stock table.
    rows = [sep_row("AAA", "2024-01-02", 10.0)]  # SYNTHETIC
    panel = _build(tmp_path, rows, [tickers_row("stocks", 101, "AAA")]).panel(  # SYNTHETIC
        "2024-01-01", "2024-01-31"
    )
    assert panel.symbol.values.tolist() == [101]
