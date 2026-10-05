"""Sharadar EVENTS: one boolean variable per 8-K event code, set on the filing date.

A `FakeVendor` plays Sharadar; the real client, raw tier and dataset run
against it. EVENTS' columns, the `EVENTCODES` label and the code titles are
VERBATIM; every row value is invented (`# SYNTHETIC`). The calendar is SEP's
trading days, here the weekdays of 2024-01-02..2024-01-12.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from tests.sharadar_fixtures import (
    EVENTS_COLUMNS,
    INDICATORS_COLUMNS,
    SEP_COLUMNS,
    TICKERS_COLUMNS,
    FakeVendor,
    eventcodes_rows,
    events_row,
    sep_row,
    tickers_row,
)

DAYS = [
    "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
    "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12",
]
PERMATICKERS = {"AAA": 101, "BBB": 202}  # SYNTHETIC
CODES = {  # VERBATIM codes and titles of Sharadar's EVENTCODES list
    "22": "Results of Operations and Financial Condition",
    "52": "Departure of Directors or Certain Officers; Election of Directors",
    "91": "Financial Statements and Exhibits",
}


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("SHARADAR_API_KEY", "synthetic-key")  # SYNTHETIC


@pytest.fixture
def today(monkeypatch):
    def set_today(day: str) -> None:
        monkeypatch.setattr(
            "quantlab.acquisition.sharadar.client.vendor_today",
            lambda: date.fromisoformat(day),
        )

    set_today("2024-01-08")
    return set_today


def _vendor(events, days=DAYS[:5], tickers=("AAA",)):
    return FakeVendor(
        {
            "stocks": (SEP_COLUMNS, [sep_row(t, d, 10.0) for t in tickers for d in days]),
            "tickers": (
                TICKERS_COLUMNS,
                [tickers_row(label, PERMATICKERS[t], t) for t in tickers for label in ("SEP", "SF1")],
            ),
            "events": (EVENTS_COLUMNS, events),
            "descriptions": (INDICATORS_COLUMNS, eventcodes_rows(CODES)),
        }
    )


def _client(vendor):
    from quantlab.acquisition.sharadar.client import SharadarClient

    return SharadarClient(transport=vendor, sleep=lambda seconds: None)


def _bulk(vendor, download_dir):
    client = _client(vendor)
    for code in ("tickers", "indicators", "sep", "events"):
        client.bulk_table(code, download_dir)
    return download_dir / "sharadar"


def _dataset(tmp_path, root, **fields):
    from quantlab.dataset.config import SharadarEventsConfig
    from quantlab.dataset.sharadar.events import SharadarEventsDataset

    return SharadarEventsDataset(
        SharadarEventsConfig(
            zarr_file_path=str(tmp_path / "sharadar_events_1d.zarr"),
            raw_data_dir_path=str(root),
            **fields,
        )
    )


def _panel(tmp_path, root, **fields):
    _dataset(tmp_path, root, **fields).update()
    return _dataset(tmp_path, root, **fields).panel("2024-01-01", "2024-12-31").load()


def _set_days(panel, variable, permaticker=101) -> list[str]:
    column = panel[variable].sel(symbol=permaticker).values
    stamps = pd.DatetimeIndex(panel["timestamp"].values).strftime("%Y-%m-%d")
    return [day for day, flag in zip(stamps, column, strict=True) if flag]


def test_a_pipe_joined_event_list_sets_each_of_its_codes_on_the_filing_date(tmp_path, today):
    events = [events_row("AAA", "2024-01-03", "22|91"), events_row("AAA", "2024-01-05", "52")]  # SYNTHETIC
    panel = _panel(tmp_path, _bulk(_vendor(events), tmp_path / "dl"))

    assert sorted(panel.data_vars) == ["event_22", "event_52", "event_91"]
    assert panel["event_22"].dtype == bool
    assert _set_days(panel, "event_22") == ["2024-01-03"]
    assert _set_days(panel, "event_91") == ["2024-01-03"]
    assert _set_days(panel, "event_52") == ["2024-01-05"]
    assert panel["event_22"].attrs["title"] == CODES["22"]
    assert panel["symbol"].values.tolist() == [101]
    assert pd.DatetimeIndex(panel["timestamp"].values).strftime("%Y-%m-%d").tolist() == DAYS[:5]


def test_a_filing_on_a_weekend_is_set_on_the_next_trading_day(tmp_path, today):
    events = [events_row("AAA", "2024-01-06", "22")]  # SYNTHETIC: a Saturday
    panel = _panel(tmp_path, _bulk(_vendor(events), tmp_path / "dl"))

    assert _set_days(panel, "event_22") == ["2024-01-08"]


def test_the_event_codes_come_from_the_published_list(tmp_path, today):
    # Every code of the list is a variable, even one no filing used yet.
    events = [events_row("AAA", "2024-01-03", "22")]  # SYNTHETIC
    panel = _panel(tmp_path, _bulk(_vendor(events), tmp_path / "dl"))
    assert not panel["event_52"].values.any()

    # A code the list does not publish is refused rather than dropped.
    unknown = [events_row("AAA", "2024-01-03", "22|99")]  # SYNTHETIC
    root = _bulk(_vendor(unknown), tmp_path / "unknown")
    with pytest.raises(ValueError, match="99"):
        _dataset(tmp_path / "unknown", root).update()


def test_an_update_appends_new_filings_and_keeps_stored_days(tmp_path, today):
    events = [events_row("AAA", "2024-01-03", "22")]  # SYNTHETIC
    vendor = _vendor(events, days=DAYS, tickers=("AAA", "BBB"))
    columns, sep = vendor.tables["stocks"]
    vendor.tables["stocks"] = (columns, [r for r in sep if r["date"] <= "2024-01-08"])
    root = _bulk(vendor, tmp_path / "dl")
    _dataset(tmp_path, root).update()
    before = _dataset(tmp_path, root).panel("2024-01-01", "2024-12-31").load()

    vendor.tables["stocks"] = (columns, sep)
    events.append(events_row("BBB", "2024-01-10", "91"))  # SYNTHETIC
    today("2024-01-12")
    client = _client(vendor)
    client.window_table("sep", tmp_path / "dl")
    client.window_table("events", tmp_path / "dl")
    _dataset(tmp_path, root).update()

    after = _dataset(tmp_path, root).panel("2024-01-01", "2024-12-31").load()
    assert pd.DatetimeIndex(after["timestamp"].values).strftime("%Y-%m-%d").tolist() == DAYS
    assert after["symbol"].values.tolist() == [101, 202]
    assert after.sel(timestamp=slice(None, "2024-01-08"), symbol=[101]).identical(before)
    assert _set_days(after, "event_91", 202) == ["2024-01-10"]
    assert not np.asarray(after["event_91"].sel(symbol=202).values[:5]).any()


def test_a_filing_before_the_first_trading_day_is_left_out(tmp_path, today):
    # EVENTS starts years before SEP's first trading day; such a filing must
    # not be rolled forward onto it.
    events = [
        events_row("AAA", "1993-11-08", "22"),  # SYNTHETIC
        events_row("AAA", "2024-01-03", "91"),  # SYNTHETIC
    ]
    panel = _panel(tmp_path, _bulk(_vendor(events), tmp_path / "dl"))

    assert _set_days(panel, "event_22") == []
    assert _set_days(panel, "event_91") == ["2024-01-03"]


def test_a_code_published_after_the_store_was_built_widens_it_false_on_stored_days(
    tmp_path, today
):
    events = [events_row("AAA", "2024-01-03", "22")]  # SYNTHETIC
    vendor = _vendor(events, days=DAYS)
    columns, sep = vendor.tables["stocks"]
    vendor.tables["stocks"] = (columns, [r for r in sep if r["date"] <= "2024-01-08"])
    root = _bulk(vendor, tmp_path / "dl")
    _dataset(tmp_path, root).update()
    stored = _dataset(tmp_path, root).panel("2024-01-01", "2024-12-31").load()

    vendor.tables["stocks"] = (columns, sep)
    vendor.tables["descriptions"][1].extend(eventcodes_rows({"99": "A new item (SYNTHETIC)"}))
    events.append(events_row("AAA", "2024-01-10", "99"))  # SYNTHETIC
    today("2024-01-12")
    client = _client(vendor)
    client.window_table("sep", tmp_path / "dl")
    client.window_table("events", tmp_path / "dl")
    client.bulk_table("indicators", tmp_path / "dl")
    _dataset(tmp_path, root).update()

    after = _dataset(tmp_path, root).panel("2024-01-01", "2024-12-31").load()
    # No filing listed the code before it was published: False on stored days.
    assert _set_days(after, "event_99") == ["2024-01-10"]
    assert after[list(stored.data_vars)].sel(timestamp=slice(None, "2024-01-08")).identical(stored)
