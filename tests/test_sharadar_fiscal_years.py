"""Sharadar fiscal-year history: the latest five SF1 ARY rows, point in time.

A `FakeVendor` plays Sharadar; the real client, raw tier and dataset run
against it. SF1's columns are VERBATIM, every row value is invented
(`# SYNTHETIC`). The panel's calendar is SEP's trading days, here the
weekdays of January 2024; the fiscal years are invented to be released on
them.
"""

from __future__ import annotations

import json

from datetime import date

import numpy as np
import pandas as pd
import pytest

from tests.sharadar_fixtures import (
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
PERMATICKERS = {"AAA": 101, "BBB": 202}  # SYNTHETIC

UNITS = [
    {
        "table": "SF1",
        "indicator": indicator,
        "isfilter": "N",  # SYNTHETIC
        "isprimarykey": "N",  # SYNTHETIC
        "title": indicator,  # SYNTHETIC
        "description": "Synthetic description",  # SYNTHETIC
        "unittype": "USD/share",  # SYNTHETIC
    }
    for indicator in ("eps", "sps")
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

    set_today("2024-01-17")
    return set_today


def _vendor(sf1, days=DAYS, tickers=("AAA",)):
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


def _client(vendor):
    from quantlab.acquisition.sharadar.client import SharadarClient

    return SharadarClient(transport=vendor, sleep=lambda seconds: None)


def _bulk(vendor, download_dir):
    client = _client(vendor)
    for code in ("tickers", "indicators", "sep", "sf1"):
        client.bulk_table(code, download_dir)
    return download_dir / "sharadar"


def _dataset(tmp_path, root, **fields):
    from quantlab.dataset.config import SharadarFiscalYearsConfig
    from quantlab.dataset.sharadar.fiscal_years import SharadarFiscalYearsDataset

    return SharadarFiscalYearsDataset(
        SharadarFiscalYearsConfig(
            zarr_file_path=str(tmp_path / "sharadar_sf1_fiscal_years.zarr"),
            raw_data_dir_path=str(root),
            **fields,
        )
    )


def _panel(tmp_path, root, **fields):
    """Build the store with ``update()`` and read it back."""
    _dataset(tmp_path, root, **fields).update()
    return _dataset(tmp_path, root, **fields).panel("2024-01-01", "2024-12-31").load()


def _on(panel, day, ticker="AAA", prefix="eps", years=5) -> list[float]:
    """The ``prefix_fy0..`` slots of one security on one day, NaN kept."""
    cell = panel.sel(timestamp=day, symbol=PERMATICKERS[ticker])
    return [float(cell[f"{prefix}_fy{k}"]) for k in range(years)]


def _same(actual: list[float], expected: list[float]) -> bool:
    return np.array_equal(np.asarray(actual), np.asarray(expected), equal_nan=True)


NAN = float("nan")


def _ary(ticker, released, year, eps, sps=None, **values):
    """One ARY row of fiscal year ``year`` (ending 30 September), released on ``released``."""
    return sf1_row(
        ticker, "ARY", released, f"{year}-09-30", eps=eps,
        sps=eps * 10 if sps is None else sps, **values,
    )  # SYNTHETIC


# -- placing rows at their release date --------------------------------------


def test_a_fiscal_year_appears_from_its_release_date_and_not_a_day_earlier(tmp_path, today):
    panel = _panel(tmp_path, _bulk(_vendor([_ary("AAA", "2024-01-05", 2023, 1.5)]), tmp_path / "dl"))

    assert _same(_on(panel, "2024-01-04"), [NAN] * 5)
    assert _same(_on(panel, "2024-01-05"), [1.5, NAN, NAN, NAN, NAN])
    assert _same(_on(panel, "2024-01-17"), [1.5, NAN, NAN, NAN, NAN])
    assert _same(_on(panel, "2024-01-05", prefix="sps"), [15.0, NAN, NAN, NAN, NAN])
    assert panel["symbol"].values.tolist() == [101]


def test_a_weekend_release_appears_from_the_next_trading_day(tmp_path, today):
    panel = _panel(tmp_path, _bulk(_vendor([_ary("AAA", "2024-01-06", 2023, 1.5)]), tmp_path / "dl"))

    assert np.isnan(_on(panel, "2024-01-05")[0])
    assert _on(panel, "2024-01-08")[0] == 1.5


def test_a_new_fiscal_year_shifts_the_slots(tmp_path, today):
    # Six fiscal years released one per day: the oldest falls off the fifth slot.
    sf1 = [
        _ary("AAA", day, 2018 + k, float(k + 1))
        for k, day in enumerate(DAYS[:6])
    ]
    panel = _panel(tmp_path, _bulk(_vendor(sf1), tmp_path / "dl"))

    assert _same(_on(panel, DAYS[0]), [1.0, NAN, NAN, NAN, NAN])
    assert _same(_on(panel, DAYS[1]), [2.0, 1.0, NAN, NAN, NAN])
    assert _same(_on(panel, DAYS[4]), [5.0, 4.0, 3.0, 2.0, 1.0])
    assert _same(_on(panel, DAYS[5]), [6.0, 5.0, 4.0, 3.0, 2.0])
    fiscal_year_ends = [
        pd.Timestamp(panel[f"reportperiod_fy{k}"].sel(timestamp=DAYS[5], symbol=101).values)
        for k in range(5)
    ]
    assert fiscal_year_ends == [pd.Timestamp(f"{y}-09-30") for y in (2023, 2022, 2021, 2020, 2019)]


def test_several_years_released_on_one_day_fill_their_slots_at_once(tmp_path, today):
    # A newly covered company's history arrives in one bulk: three years, one day.
    sf1 = [_ary("AAA", "2024-01-09", y, float(y - 2020)) for y in (2021, 2022, 2023)]
    panel = _panel(tmp_path, _bulk(_vendor(sf1), tmp_path / "dl"))

    assert _same(_on(panel, "2024-01-08"), [NAN] * 5)
    assert _same(_on(panel, "2024-01-09"), [3.0, 2.0, 1.0, NAN, NAN])


def test_a_company_with_fewer_years_has_nan_in_the_older_slots(tmp_path, today):
    sf1 = [_ary("AAA", "2024-01-03", 2022, 1.0), _ary("AAA", "2024-01-04", 2023, 2.0)]
    panel = _panel(tmp_path, _bulk(_vendor(sf1), tmp_path / "dl"))

    assert _same(_on(panel, "2024-01-10"), [2.0, 1.0, NAN, NAN, NAN])
    assert np.isnat(panel["reportperiod_fy2"].sel(timestamp="2024-01-10", symbol=101).values)


def test_the_number_of_slots_is_configurable(tmp_path, today):
    sf1 = [_ary("AAA", day, 2020 + k, float(k + 1)) for k, day in enumerate(DAYS[:3])]
    panel = _panel(tmp_path, _bulk(_vendor(sf1), tmp_path / "dl"), years=2, indicators=("eps",))

    assert sorted(v for v in panel.data_vars if v.startswith("eps")) == ["eps_fy0", "eps_fy1"]
    assert not any(v.startswith("sps") for v in panel.data_vars)
    assert _same(_on(panel, DAYS[2], years=2), [3.0, 2.0])


# -- restatements --------------------------------------------------------------


def test_a_restated_year_replaces_the_earlier_value_only_from_its_own_release_date(
    tmp_path, today
):
    sf1 = [
        _ary("AAA", "2024-01-03", 2022, 1.0),
        _ary("AAA", "2024-01-04", 2023, 2.0),
        # Fiscal 2022 refiled later with a corrected value: it is the fy1 slot.
        _ary("AAA", "2024-01-10", 2022, 1.1),
        # Fiscal 2023 refiled too: the fy0 slot.
        _ary("AAA", "2024-01-12", 2023, 2.2),
    ]
    panel = _panel(tmp_path, _bulk(_vendor(sf1), tmp_path / "dl"))

    assert _same(_on(panel, "2024-01-09")[:2], [2.0, 1.0])
    assert _same(_on(panel, "2024-01-10")[:2], [2.0, 1.1])
    assert _same(_on(panel, "2024-01-11")[:2], [2.0, 1.1])
    assert _same(_on(panel, "2024-01-12")[:2], [2.2, 1.1])
    release = panel["release_date"].sel(symbol=101)
    assert pd.Timestamp(release.sel(timestamp="2024-01-11").values) == pd.Timestamp("2024-01-04")
    assert pd.Timestamp(release.sel(timestamp="2024-01-12").values) == pd.Timestamp("2024-01-12")


def test_only_annual_as_reported_rows_feed_the_panel(tmp_path, today):
    sf1 = [
        sf1_row("AAA", "ARQ", "2024-01-03", "2023-06-30", eps=0.4),  # SYNTHETIC
        sf1_row("AAA", "ART", "2024-01-03", "2023-06-30", eps=1.4),  # SYNTHETIC
        # MRY is dated at its period end, before the 10-K is filed.
        sf1_row("AAA", "MRY", "2023-09-30", "2023-09-30", eps=9.9),  # SYNTHETIC
        _ary("AAA", "2024-01-10", 2023, 2.0),
    ]
    panel = _panel(tmp_path, _bulk(_vendor(sf1), tmp_path / "dl"))

    shown = panel["eps_fy0"].sel(symbol=101).values
    assert np.isnan(shown[:6]).all()
    assert set(shown[6:].tolist()) == {2.0}


def test_rows_of_one_permaticker_with_one_key_under_two_tickers_are_left_out_and_reported(
    tmp_path, today
):
    sf1 = [
        _ary("AAA", "2024-01-04", 2023, 1.0),
        _ary("AAA.OLD", "2024-01-04", 2023, 1.1),
        _ary("AAA", "2024-01-03", 2022, 0.9),
    ]
    vendor = _vendor(sf1)
    vendor.tables["tickers"][1].append(tickers_row("SF1", 101, "AAA.OLD"))  # SYNTHETIC
    root = _bulk(vendor, tmp_path / "dl")

    _dataset(tmp_path, root).update()
    (path,) = tmp_path.glob("*.unmapped.json")
    entries = {e["ticker"]: e for e in json.loads(path.read_text())["unmapped"]}
    assert sorted(entries) == ["AAA", "AAA.OLD"]
    assert entries["AAA"]["rows"] == 1


# -- staleness, units and universe ---------------------------------------------


def test_a_history_not_replaced_goes_stale(tmp_path, today):
    sf1 = [_ary("AAA", "2024-01-03", 2022, 1.0), _ary("AAA", "2024-01-04", 2023, 2.0)]
    root = _bulk(_vendor(sf1), tmp_path / "dl")

    stale = _panel(tmp_path, root, stale_after_days=4)
    forever = _panel(tmp_path / "forever", root, stale_after_days=None)

    # Counted from fiscal 2023's release: shown through 2024-01-08, all slots.
    assert _same(_on(stale, "2024-01-08")[:2], [2.0, 1.0])
    assert _same(_on(stale, "2024-01-09")[:2], [NAN, NAN])
    assert np.isnat(stale["reportperiod_fy0"].sel(timestamp="2024-01-09", symbol=101).values)
    assert _same(_on(forever, "2024-01-17")[:2], [2.0, 1.0])


def test_each_variable_records_its_vendor_unit_and_its_slot(tmp_path, today):
    panel = _panel(tmp_path, _bulk(_vendor([_ary("AAA", "2024-01-04", 2023, 1.0)]), tmp_path / "dl"))

    assert panel["eps_fy0"].attrs["unit"] == "USD/share"
    assert panel["sps_fy3"].attrs["unit"] == "USD/share"
    assert "3" in panel["sps_fy3"].attrs["description"]


def test_the_universe_rules_are_the_price_panels(tmp_path, today):
    sf1 = [_ary(t, "2024-01-04", 2023, 1.0) for t in ("AAA", "BBB")]
    vendor = _vendor(sf1, tickers=("AAA", "BBB"))
    for row in vendor.tables["tickers"][1]:
        if row["ticker"] == "BBB":
            row["category"] = "ADR Common Stock"  # SYNTHETIC
    root = _bulk(vendor, tmp_path / "dl")

    default = _panel(tmp_path, root)
    rostered = _panel(tmp_path / "rostered", root, permatickers=(202,))

    assert default["symbol"].values.tolist() == [101]
    assert rostered["symbol"].values.tolist() == [202]


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        ({"years": 0}, "years"),
        ({"indicators": ()}, "indicators"),
        ({"indicators": ("eps", "nosuch")}, "nosuch"),
        ({"indicators": ("eps", "eps")}, "indicators"),
        ({"stale_after_days": 0}, "stale_after_days"),
        ({"table": "sep"}, "sf1"),
    ],
)
def test_an_invalid_config_is_refused(tmp_path, fields, match):
    with pytest.raises(ValueError, match=match):
        _dataset(tmp_path, tmp_path, **fields)


# -- updates -------------------------------------------------------------------


def test_an_update_appends_new_years_and_never_rewrites_stored_days(tmp_path, today):
    sf1 = [_ary("AAA", "2024-01-04", 2022, 1.0)]
    vendor = _vendor(sf1, tickers=("AAA", "BBB"))
    columns, sep = vendor.tables["stocks"]
    vendor.tables["stocks"] = (columns, [r for r in sep if r["date"] <= "2024-01-11"])
    today("2024-01-11")
    root = _bulk(vendor, tmp_path / "dl")
    _dataset(tmp_path, root).update()
    before = _dataset(tmp_path, root).panel("2024-01-01", "2024-01-11").load()

    vendor.tables["stocks"] = (columns, sep)
    sf1[0] = _ary("AAA", "2024-01-04", 2022, 1.05, lastupdated="2024-01-15")
    sf1.append(_ary("AAA", "2024-01-16", 2023, 2.0, lastupdated="2024-01-16"))
    sf1.append(_ary("BBB", "2024-01-16", 2023, 7.0, lastupdated="2024-01-16"))
    today("2024-01-17")
    client = _client(vendor)
    client.window_table("sep", tmp_path / "dl", through="2024-01-17")
    client.updated_table("sf1", tmp_path / "dl")
    _dataset(tmp_path, root).update()

    after = _dataset(tmp_path, root).panel("2024-01-01", "2024-12-31").load()
    assert pd.DatetimeIndex(after["timestamp"].values).strftime("%Y-%m-%d").tolist() == DAYS
    assert after.sel(timestamp=slice(None, "2024-01-11"), symbol=[101]).identical(before)
    assert after["symbol"].values.tolist() == [101, 202]
    assert _same(_on(after, "2024-01-15")[:2], [1.05, NAN])
    assert _same(_on(after, "2024-01-16")[:2], [2.0, 1.05])
    assert _same(_on(after, "2024-01-16", "BBB")[:1], [7.0])
    assert np.isnat(after["release_date"].sel(timestamp="2024-01-11", symbol=202).values)


def test_the_config_round_trips_through_json(tmp_path):
    import json

    from quantlab.dataset.sharadar.fiscal_years import SharadarFiscalYearsDataset

    config = _dataset(tmp_path, tmp_path / "raw", indicators=("sps", "netinccmn"), years=3).config
    saved = json.loads(json.dumps(config.to_dict()))

    assert SharadarFiscalYearsDataset.from_config(saved).config == config
