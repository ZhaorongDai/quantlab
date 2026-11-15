"""The CIK map: a company panel keyed by SEC CIK read on Sharadar's permatickers.

The company panel is StockFit's quarterly statements panel, built from a raw
tier the refresh entry wrote from the recorded responses (VERBATIM, see
``tests/test_stockfit_statements.py``): PLUG's 2026-09-30 quarter (revenue
200,602,000) shows from 2026-11-10, the quarter before it until then.
Sharadar's TICKERS is played by a ``FakeVendor``; its permatickers, price
dates and the second share class are invented (`# SYNTHETIC`), the CIKs are
PLUG's and TWTR's own.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tests.sharadar_fixtures import TICKERS_COLUMNS, FakeVendor, tickers_row
from tests.stockfit_fixtures import PLUG, TWTR
from tests.test_stockfit_statements import _raw_tier

PLUG_A, PLUG_B, TWTR_SEC, OTHER = 101, 102, 303, 404  # SYNTHETIC permatickers; PLUG_B is PLUG's second class
LATEST = 200_602_000  # PLUG's 2026-09-30 revenue, shown from 2026-11-10


def _filings(cik: int) -> str:
    return f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik:010d}"


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("STOCKFIT_API_KEY", "synthetic-stockfit-key")  # SYNTHETIC
    monkeypatch.setenv("SHARADAR_API_KEY", "synthetic-key")  # SYNTHETIC
    monkeypatch.setattr("quantlab.acquisition.sharadar.client.vendor_today", lambda: date(2026, 11, 16))


def _sharadar(tmp_path, rows):
    from quantlab.acquisition.sharadar.client import SharadarClient

    client = SharadarClient(transport=FakeVendor({"tickers": (TICKERS_COLUMNS, rows)}), sleep=lambda seconds: None)
    client.bulk_table("tickers", tmp_path / "downloads")
    return tmp_path / "downloads" / "sharadar"


def _tickers(**plug_b):
    second = {"firstpricedate": "2010-01-04", "lastpricedate": None, **plug_b}
    return [
        tickers_row("SEP", PLUG_A, "PLUG", secfilings=_filings(PLUG), firstpricedate="2000-01-03", lastpricedate=None),
        tickers_row("SEP", PLUG_B, "PLUG.B", category="Domestic Common Stock Secondary Class",
                    secfilings=_filings(PLUG), **second),
        tickers_row("SEP", TWTR_SEC, "TWTR", secfilings=_filings(TWTR), lastpricedate=None),
        tickers_row("SEP", OTHER, "OTHER", secfilings=_filings(1)),  # a CIK the company panel lacks
        tickers_row("SF1", PLUG_A, "PLUG", secfilings=_filings(PLUG)),  # not a SEP row: no security of its own
    ]


@pytest.fixture
def statements(env, tmp_path):
    from quantlab.dataset.config import StockFitStatementsConfig
    from quantlab.dataset.stockfit.statements import StockFitStatementsDataset

    config = StockFitStatementsConfig(
        zarr_file_path=str(tmp_path / "statements.zarr"),
        raw_data_dir_path=str(_raw_tier(tmp_path)),
        start_date="2026-01-01",
    )
    return StockFitStatementsDataset(config).update()


def _view(tmp_path, company, **plug_b):
    from quantlab.dataset.sharadar.cik_map import CikMapDataset

    return CikMapDataset(company, _sharadar(tmp_path, _tickers(**plug_b)))


def test_both_share_classes_of_a_two_class_company_carry_its_statement_values(tmp_path, statements):
    panel = _view(tmp_path, statements).panel("2026-11-09", "2026-11-13").load()

    for security in (PLUG_A, PLUG_B):
        revenue = panel["revenue"].sel(symbol=security)
        assert revenue.sel(timestamp="2026-11-10").item() == LATEST
        assert revenue.sel(timestamp="2026-11-09").item() != LATEST
    assert panel["period_end"].sel(symbol=PLUG_B, timestamp="2026-11-10").item() == pd.Timestamp("2026-09-30").value


@pytest.mark.parametrize(
    ("first", "last", "shown"),
    [
        ("2010-01-04", "2026-11-10", ["2026-11-10"]),  # SYNTHETIC: delisted after 2026-11-10
        ("2026-11-12", None, ["2026-11-12", "2026-11-13"]),  # SYNTHETIC: first priced 2026-11-12
    ],
)
def test_a_cik_with_no_security_on_a_day_contributes_nothing_that_day(tmp_path, statements, first, last, shown):
    panel = _view(tmp_path, statements, firstpricedate=first, lastpricedate=last).panel("2026-11-10", "2026-11-13")

    second = panel["revenue"].sel(symbol=PLUG_B).to_series()
    assert second.dropna().index.strftime("%Y-%m-%d").tolist() == shown
    assert (second.dropna() == LATEST).all()
    assert panel["period_end"].sel(symbol=PLUG_B).to_series().dropna().index.strftime("%Y-%m-%d").tolist() == shown
    assert (panel["revenue"].sel(symbol=PLUG_A) == LATEST).all()  # the other class is priced throughout


def test_only_securities_whose_cik_the_company_panel_holds_are_on_the_axis(tmp_path, statements):
    view = _view(tmp_path, statements)

    assert view.stored_symbols() == [PLUG_A, PLUG_B, TWTR_SEC]  # OTHER's CIK has no statements; SF1 rows are no security
    assert view.panel("2026-11-10", "2026-11-13")["symbol"].values.tolist() == [PLUG_A, PLUG_B, TWTR_SEC]
    narrow = view.panel("2026-11-10", "2026-11-13", symbols=[PLUG_B, TWTR_SEC], variables=["revenue"])
    assert narrow["symbol"].values.tolist() == [PLUG_B, TWTR_SEC] and list(narrow.data_vars) == ["revenue"]
    with pytest.raises(KeyError, match=r"\[404\]"):
        view.panel("2026-11-10", "2026-11-13", symbols=[OTHER])


def test_any_company_panel_on_the_cik_axis_maps_the_same_way(env, tmp_path):
    from quantlab.dataset.memory import FrameDataset
    from quantlab.dataset.sharadar.cik_map import CikMapDataset

    days = pd.bdate_range("2026-11-09", "2026-11-13")
    filings = xr.Dataset(  # SYNTHETIC: one filing count per company and day
        {"filings": (("timestamp", "symbol"), np.arange(10, dtype=float).reshape(5, 2))},
        coords={"timestamp": days, "symbol": np.array([PLUG, TWTR], dtype=np.int64)},
    )
    raw = _sharadar(tmp_path, _tickers(lastpricedate="2026-11-11"))
    company = FrameDataset(filings).to_zarr(tmp_path / "filings.zarr")
    panel = CikMapDataset(company, raw).panel("2026-11-09", "2026-11-13")

    assert panel["filings"].sel(symbol=PLUG_A).values.tolist() == [0.0, 2.0, 4.0, 6.0, 8.0]
    assert panel["filings"].sel(symbol=PLUG_B).values[:3].tolist() == [0.0, 2.0, 4.0]
    assert np.isnan(panel["filings"].sel(symbol=PLUG_B).values[3:]).all()
    assert panel["filings"].sel(symbol=TWTR_SEC).values.tolist() == [1.0, 3.0, 5.0, 7.0, 9.0]


def test_the_view_rebuilds_from_its_config_and_records_the_map(tmp_path, statements):
    from quantlab.core.component import rebuild
    from quantlab.runs.record import DataRecorder

    view = _view(tmp_path, statements)
    again = rebuild(view.get_config())
    xr.testing.assert_identical(again.panel("2026-11-10", "2026-11-13"), view.panel("2026-11-10", "2026-11-13"))
    with DataRecorder(keys=[(statements, "statements"), (view, "cik_map")]) as recorder:
        view.panel("2026-11-10", "2026-11-13")
    assert set(recorder.records) == {"statements", "cik_map.cik_map"}


def test_head_repeats_a_company_row_on_each_of_its_securities(tmp_path, statements):
    view = _view(tmp_path, statements)

    assert statements.head(2).collect()["symbol"].to_list() == [PLUG, TWTR]
    assert view.head(2).collect()["symbol"].to_list() == [PLUG_A, PLUG_B, TWTR_SEC]
