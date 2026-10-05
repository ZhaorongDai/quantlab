"""Sharadar point-in-time industry: each permaticker's SIC on each day, as a Fama-French 48 code.

A `FakeVendor` plays Sharadar; the real client, raw tier and dataset run
against it. The `sicchangefrom`/`sicchangeto` action types and the TICKERS
columns are VERBATIM; every row value is invented (`# SYNTHETIC`). The SIC
codes are real codes so their Fama-French 48 industries are known: 3571
(electronic computers) is 35 "Comps", 6021 (national commercial banks) is 44
"Banks", 2834 (pharmaceutical preparations) is 13 "Drugs", and 9995
(non-operating establishments) is inside no range. The calendar is SEP's
trading days, the weekdays of 2024-01-02..2024-01-12.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
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

DAYS = [
    "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
    "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12",
]
PERMATICKERS = {"AAA": 101, "BBB": 202, "CCC": 303}  # SYNTHETIC
COMPS, DRUGS, BANKS, OTHER = 35.0, 13.0, 44.0, 48.0


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

    set_today("2024-01-12")
    return set_today


def _sic_change(day: str, ticker: str, old: int, new: int) -> list[dict]:
    """The pair of ACTIONS rows Sharadar writes for one SIC change."""
    return [
        action_row(day, "sicchangefrom", ticker, float(old)),  # SYNTHETIC
        action_row(day, "sicchangeto", ticker, float(new)),  # SYNTHETIC
    ]


def _vendor(sic: dict[str, int | None], actions: list[dict], days=DAYS, **tickers_fields):
    return FakeVendor(
        {
            "stocks": (SEP_COLUMNS, [sep_row(t, d, 10.0) for t in sic for d in days]),
            "tickers": (
                TICKERS_COLUMNS,
                [
                    tickers_row(
                        "SEP", PERMATICKERS[t], t, siccode=code, **tickers_fields.get(t, {})
                    )
                    for t, code in sic.items()
                ],
            ),
            "actions": (ACTIONS_COLUMNS, actions),
        }
    )


def _client(vendor):
    from quantlab.acquisition.sharadar.client import SharadarClient

    return SharadarClient(transport=vendor, sleep=lambda seconds: None)


def _bulk(vendor, download_dir):
    client = _client(vendor)
    for code in ("tickers", "sep", "actions"):
        client.bulk_table(code, download_dir)
    return download_dir / "sharadar"


def _dataset(tmp_path, root, **fields):
    from quantlab.dataset.config import SharadarIndustryConfig
    from quantlab.dataset.sharadar.industry import SharadarIndustryDataset

    fields.setdefault("industry_merge", ())
    return SharadarIndustryDataset(
        SharadarIndustryConfig(
            zarr_file_path=str(tmp_path / "sharadar_industry_1d.zarr"),
            raw_data_dir_path=str(root),
            **fields,
        )
    )


def _panel(tmp_path, root, **fields):
    _dataset(tmp_path, root, **fields).update()
    return _dataset(tmp_path, root, **fields).panel("2024-01-01", "2024-12-31").load()


def _industry(panel, permaticker) -> list[float]:
    return panel["industry"].sel(symbol=permaticker).values.tolist()


def test_an_sic_change_shows_the_old_industry_before_its_action_date_and_the_new_one_from_it(
    tmp_path, today
):
    actions = _sic_change("2024-01-05", "AAA", 3571, 6021)  # SYNTHETIC
    panel = _panel(tmp_path, _bulk(_vendor({"AAA": 6021}, actions), tmp_path / "dl"))

    assert list(panel.data_vars) == ["industry"]
    assert panel["symbol"].values.tolist() == [101]
    assert pd.DatetimeIndex(panel["timestamp"].values).strftime("%Y-%m-%d").tolist() == DAYS
    assert _industry(panel, 101) == [COMPS] * 3 + [BANKS] * 6


def test_successive_changes_walk_back_from_the_current_sic_and_a_weekend_change_starts_monday(
    tmp_path, today
):
    actions = [
        *_sic_change("2024-01-04", "AAA", 2834, 3571),  # SYNTHETIC
        *_sic_change("2024-01-06", "AAA", 3571, 6021),  # SYNTHETIC: a Saturday
    ]
    panel = _panel(tmp_path, _bulk(_vendor({"AAA": 6021}, actions), tmp_path / "dl"))

    assert _industry(panel, 101) == [DRUGS] * 2 + [COMPS] * 2 + [BANKS] * 5


def test_a_symbol_with_no_sic_change_keeps_its_current_industry_over_its_whole_history(
    tmp_path, today
):
    actions = _sic_change("2024-01-05", "AAA", 3571, 6021)  # SYNTHETIC
    panel = _panel(tmp_path, _bulk(_vendor({"AAA": 6021, "BBB": 2834}, actions), tmp_path / "dl"))

    assert _industry(panel, 202) == [DRUGS] * len(DAYS)


def test_an_sic_outside_every_range_is_other_and_an_unknown_sic_is_nan(tmp_path, today):
    panel = _panel(
        tmp_path, _bulk(_vendor({"AAA": 9995, "BBB": None}, []), tmp_path / "dl")
    )

    assert _industry(panel, 101) == [OTHER] * len(DAYS)
    assert np.isnan(_industry(panel, 202)).all()


def test_the_industry_is_shown_only_between_the_first_and_last_price_dates(tmp_path, today):
    vendor = _vendor(
        {"AAA": 3571},
        [],
        AAA={"firstpricedate": "2024-01-04", "lastpricedate": "2024-01-10"},  # SYNTHETIC
    )
    panel = _panel(tmp_path, _bulk(vendor, tmp_path / "dl"))

    assert np.isnan(_industry(panel, 101)[:2]).all()
    assert _industry(panel, 101)[2:7] == [COMPS] * 5
    assert np.isnan(_industry(panel, 101)[7:]).all()


def test_the_configured_merge_mapping_is_applied(tmp_path, today):
    actions = _sic_change("2024-01-05", "AAA", 3571, 6021)  # SYNTHETIC
    root = _bulk(_vendor({"AAA": 6021, "BBB": 2834}, actions), tmp_path / "dl")
    panel = _panel(tmp_path, root, industry_merge=((35, 44),))

    assert _industry(panel, 101) == [BANKS] * len(DAYS)
    assert _industry(panel, 202) == [DRUGS] * len(DAYS)


@pytest.mark.parametrize(
    "merge, message",
    [
        (((35, 49),), r"unknown code\(s\) \[49\]"),
        (((0, 44),), r"unknown code\(s\) \[0\]"),
        (((35, 44), (44, 13)), "themselves merged away"),
        (((35, 44), (35, 13)), "two targets"),
        (((35, 35),), "into themselves"),
    ],
)
def test_an_invalid_merge_mapping_is_refused(tmp_path, merge, message):
    with pytest.raises(ValueError, match=message):
        _dataset(tmp_path, tmp_path / "dl" / "sharadar", industry_merge=merge)


def test_the_default_merge_mapping_is_valid_and_is_the_config_default():
    from quantlab.dataset._support.ff48 import check_merge
    from quantlab.dataset.config import DEFAULT_INDUSTRY_MERGE, SharadarIndustryConfig

    assert DEFAULT_INDUSTRY_MERGE
    assert check_merge(DEFAULT_INDUSTRY_MERGE, "test") == DEFAULT_INDUSTRY_MERGE
    config = SharadarIndustryConfig(zarr_file_path="x.zarr", raw_data_dir_path="raw")
    assert config.industry_merge == DEFAULT_INDUSTRY_MERGE


def test_the_config_round_trips_through_json(tmp_path):
    import json

    from quantlab.dataset.sharadar.industry import SharadarIndustryDataset

    config = _dataset(tmp_path, tmp_path / "raw", industry_merge=((35, 44), (13, 44))).config
    saved = json.loads(json.dumps(config.to_dict()))

    assert config.industry_merge == ((13, 44), (35, 44))
    assert SharadarIndustryDataset.from_config(saved).config == config


def test_a_ticker_outside_sep_is_another_security_and_its_changes_are_dropped(tmp_path, today):
    actions = [
        *_sic_change("2024-01-05", "AAA", 3571, 6021),  # SYNTHETIC
        *_sic_change("2024-01-05", "ZZZ", 3571, 6021),  # SYNTHETIC: not a SEP ticker
    ]
    panel = _panel(tmp_path, _bulk(_vendor({"AAA": 6021}, actions), tmp_path / "dl"))

    assert panel["symbol"].values.tolist() == [101]


def test_two_changes_of_one_security_on_one_day_are_refused(tmp_path, today):
    actions = [
        *_sic_change("2024-01-05", "AAA", 3571, 6021),  # SYNTHETIC
        *_sic_change("2024-01-05", "AAA", 2834, 6021),  # SYNTHETIC
    ]
    root = _bulk(_vendor({"AAA": 6021}, actions), tmp_path / "dl")
    with pytest.raises(ValueError, match="several SIC changes"):
        _dataset(tmp_path, root).update()


def test_an_update_appends_new_days_and_keeps_stored_ones(tmp_path, today):
    vendor = _vendor({"AAA": 3571, "BBB": 2834}, [])
    columns, sep = vendor.tables["stocks"]
    vendor.tables["stocks"] = (columns, [r for r in sep if r["date"] <= "2024-01-08"])
    today("2024-01-08")
    root = _bulk(vendor, tmp_path / "dl")
    _dataset(tmp_path, root).update()
    before = _dataset(tmp_path, root).panel("2024-01-01", "2024-12-31").load()

    # On 2024-01-10 AAA's SIC changes: TICKERS shows the new code, ACTIONS the change.
    vendor.tables["stocks"] = (columns, sep)
    vendor.tables["tickers"][1][0]["siccode"] = 6021  # SYNTHETIC
    vendor.tables["actions"][1].extend(_sic_change("2024-01-10", "AAA", 3571, 6021))
    today("2024-01-12")
    client = _client(vendor)
    client.bulk_table("tickers", tmp_path / "dl")
    client.window_table("sep", tmp_path / "dl")
    client.window_table("actions", tmp_path / "dl")
    _dataset(tmp_path, root).update()

    after = _dataset(tmp_path, root).panel("2024-01-01", "2024-12-31").load()
    assert pd.DatetimeIndex(after["timestamp"].values).strftime("%Y-%m-%d").tolist() == DAYS
    assert after.sel(timestamp=slice(None, "2024-01-08")).identical(before)
    assert _industry(after, 101) == [COMPS] * 6 + [BANKS] * 3
    assert _industry(after, 202) == [DRUGS] * len(DAYS)


def test_a_change_dated_after_the_actions_watermark_is_still_walked_back_through(tmp_path, today):
    # TICKERS is pulled whole on every run, so its current code can already
    # hold a change whose ACTIONS window has not been read into the panel's days.
    actions = _sic_change("2024-01-10", "AAA", 3571, 6021)  # SYNTHETIC
    vendor = _vendor({"AAA": 6021}, actions)
    columns, sep = vendor.tables["stocks"]
    vendor.tables["stocks"] = (columns, [r for r in sep if r["date"] <= "2024-01-08"])
    today("2024-01-08")
    panel = _panel(tmp_path, _bulk(vendor, tmp_path / "dl"))

    assert _industry(panel, 101) == [COMPS] * 5


def test_the_ff48_table_matches_frenchs_published_ranges_at_its_edges():
    # VERBATIM ranges of Siccodes48.txt (Kenneth French's data library, 2020-01-08).
    from quantlab.dataset._support.ff48 import FF48_CODES, ff48_codes

    assert FF48_CODES == tuple(range(1, 49))
    sic = np.array([100.0, 199.0, 2048.0, 2834.0, 3570.0, 3579.0, 6000.0, 6199.0, 4950.0, 4991.0, 200.0])
    assert ff48_codes(sic).tolist() == [1.0, 1.0, 1.0, 13.0, 35.0, 35.0, 44.0, 44.0, 48.0, 48.0, 1.0]


def test_no_sic_code_is_in_two_ff48_industries():
    from quantlab.dataset._support.ff48 import FF48_INDUSTRIES

    owners: dict[int, int] = {}
    for industry in FF48_INDUSTRIES:
        for low, high in industry.sic_ranges:
            for sic in range(low, high + 1):
                assert owners.setdefault(sic, industry.code) == industry.code, sic
