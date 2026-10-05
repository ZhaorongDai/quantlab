"""Sharadar share classes: the security whose filings carry each security's firm values.

A `FakeVendor` plays Sharadar; the real client, raw tier and dataset run
against it. The TICKERS columns and category values are VERBATIM; every
row value is invented (`# SYNTHETIC`). A secondary share class has SEP
rows only; its firm's DAILY market cap and SF1 fundamentals sit on the
SF1 security with the same SEC CIK. The calendar is SEP's trading days,
the weekdays of 2024-01-02..2024-01-12.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from tests.sharadar_fixtures import (
    SEP_COLUMNS,
    TICKERS_COLUMNS,
    FakeVendor,
    sep_row,
    tickers_row,
)

DAYS = [
    "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
    "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12",
]
PRIMARY = "Domestic Common Stock Primary Class"
SECONDARY = "Domestic Common Stock Secondary Class"
COMMON = "Domestic Common Stock"


def _filings(cik: str | None) -> str | None:
    """A TICKERS ``secfilings`` URL as Sharadar writes it."""
    if cik is None:
        return None
    return f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}"


#: ticker -> (permaticker, category, CIK, tables it has TICKERS rows in, overrides)
SECURITIES = {
    "AAAA": (101, PRIMARY, "0000000101", ("SEP", "SF1"), {}),  # SYNTHETIC
    "AAAB": (102, SECONDARY, "0000000101", ("SEP",), {}),  # SYNTHETIC
    "CCC": (303, COMMON, "0000000303", ("SEP", "SF1"), {}),  # SYNTHETIC
    "NOCIK": (404, SECONDARY, None, ("SEP",), {}),  # SYNTHETIC
    "ORPH": (505, SECONDARY, "0000000999", ("SEP",), {}),  # SYNTHETIC: no SF1 issuer
    # One CIK, two issuers over time (a restructuring): the secondary class
    # follows whichever is priced on the day.
    "OLD": (606, COMMON, "0000000606", ("SF1",),
            {"firstpricedate": "2020-01-02", "lastpricedate": "2024-01-04"}),  # SYNTHETIC
    "NEW": (607, PRIMARY, "0000000606", ("SF1",),
            {"firstpricedate": "2024-01-09", "lastpricedate": "2026-08-10"}),  # SYNTHETIC
    "SPLT": (608, SECONDARY, "0000000606", ("SEP",), {}),  # SYNTHETIC
    # One CIK, two issuers priced at once: no firm.
    "TWO1": (701, PRIMARY, "0000000701", ("SF1",), {}),  # SYNTHETIC
    "TWO2": (702, PRIMARY, "0000000701", ("SF1",), {}),  # SYNTHETIC
    "TWOS": (703, SECONDARY, "0000000701", ("SEP",), {}),  # SYNTHETIC
    # Three issuers, the long first one overlapping the third (not just its
    # neighbour) on 2024-01-09..10: no firm on those days only.
    "LONG": (901, COMMON, "0000000901", ("SF1",),
             {"firstpricedate": "2020-01-02", "lastpricedate": "2024-01-10"}),  # SYNTHETIC
    "SHORT": (902, COMMON, "0000000901", ("SF1",),
              {"firstpricedate": "2020-01-02", "lastpricedate": "2021-01-04"}),  # SYNTHETIC
    "THIRD": (903, PRIMARY, "0000000901", ("SF1",),
              {"firstpricedate": "2024-01-09", "lastpricedate": "2026-08-10"}),  # SYNTHETIC
    "THREES": (904, SECONDARY, "0000000901", ("SEP",), {}),  # SYNTHETIC
    # A secondary class listed after its firm: shown from its own first price.
    "LATE": (801, SECONDARY, "0000000101", ("SEP",),
             {"firstpricedate": "2024-01-08"}),  # SYNTHETIC
}


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("SHARADAR_API_KEY", "synthetic-key")  # SYNTHETIC


@pytest.fixture(autouse=True)
def _today(monkeypatch):
    monkeypatch.setattr(
        "quantlab.acquisition.sharadar.client.vendor_today", lambda: date(2024, 1, 12)
    )


def _vendor(securities=SECURITIES):
    sep_tickers = [t for t, (*_, tables, _o) in securities.items() if "SEP" in tables]
    rows = [
        tickers_row(
            table, permaticker, ticker, category=category, secfilings=_filings(cik), **overrides
        )
        for ticker, (permaticker, category, cik, tables, overrides) in securities.items()
        for table in tables
    ]
    return FakeVendor({
        "stocks": (SEP_COLUMNS, [sep_row(t, d, 10.0) for t in sep_tickers for d in DAYS]),
        "tickers": (TICKERS_COLUMNS, rows),
    })


def _root(tmp_path, vendor=None):
    from quantlab.acquisition.sharadar.client import SharadarClient

    client = SharadarClient(transport=vendor or _vendor(), sleep=lambda seconds: None)
    for code in ("tickers", "sep"):
        client.bulk_table(code, tmp_path / "downloads")
    return tmp_path / "downloads" / "sharadar"


def _dataset(tmp_path, root, **fields):
    from quantlab.dataset.config import SharadarShareClassConfig
    from quantlab.dataset.sharadar.share_class import SharadarShareClassDataset

    return SharadarShareClassDataset(SharadarShareClassConfig(
        zarr_file_path=str(tmp_path / "sharadar_share_class_1d.zarr"),
        raw_data_dir_path=str(root),
        **fields,
    ))


@pytest.fixture
def panel(tmp_path):
    root = _root(tmp_path)
    _dataset(tmp_path, root).update()
    return _dataset(tmp_path, root).panel("2024-01-01", "2024-12-31").load()


def _firm(panel, permaticker) -> list[float]:
    return panel["firm"].sel(symbol=permaticker).values.tolist()


def test_a_secondary_class_carries_the_sf1_security_with_its_cik(panel):
    assert _firm(panel, 102) == [101.0] * len(DAYS)


def test_every_other_security_is_its_own_firm(panel):
    assert _firm(panel, 101) == [101.0] * len(DAYS)
    assert _firm(panel, 303) == [303.0] * len(DAYS)


def test_a_secondary_without_a_cik_or_an_sf1_issuer_has_no_firm(panel):
    assert np.isnan(_firm(panel, 404)).all()
    assert np.isnan(_firm(panel, 505)).all()


def test_a_cik_with_successive_issuers_follows_the_one_priced_on_the_day(panel):
    # OLD priced through 2024-01-04, NEW from 2024-01-09; neither in between.
    nan = float("nan")
    want = [606.0, 606.0, 606.0, nan, nan, 607.0, 607.0, 607.0, 607.0]
    np.testing.assert_array_equal(_firm(panel, 608), want)


def test_a_cik_with_two_issuers_priced_at_once_gives_no_firm_on_those_days(panel):
    assert np.isnan(_firm(panel, 703)).all()  # both priced on every day
    # LONG alone until THIRD lists, both on 2024-01-09..10, THIRD alone after
    # LONG's last price; SHORT ended in 2021 and overlapped LONG only then.
    nan = float("nan")
    want = [901.0] * 5 + [nan, nan] + [903.0] * 2
    np.testing.assert_array_equal(_firm(panel, 904), want)


def test_the_firm_is_shown_only_from_the_securitys_own_first_price_date(panel):
    nan = float("nan")
    np.testing.assert_array_equal(_firm(panel, 801), [nan] * 4 + [101.0] * 5)


def test_only_sep_securities_are_on_the_axis(panel):
    assert sorted(panel["symbol"].values.tolist()) == [101, 102, 303, 404, 505, 608, 703, 801, 904]


def test_the_config_round_trips_through_json(tmp_path):
    import json

    from quantlab.dataset.sharadar.share_class import SharadarShareClassDataset

    config = _dataset(tmp_path, tmp_path / "raw").config
    saved = json.loads(json.dumps(config.to_dict()))
    assert SharadarShareClassDataset.from_config(saved).config == config
